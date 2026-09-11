"""
Plant model: source, transformer, cables, motors, bus and feeder breaker.

Sign and unit conventions used everywhere in this package
--------------------------------------------------------
* All impedances are COMPLEX OHMS PER PHASE, star-equivalent, referred to the
  415 V side. Inductive reactance is +j.
* All currents are line currents in amperes, rms symmetrical unless the name
  says otherwise.
* All voltages are line-to-line in volts unless the name says otherwise.
* Powers are three-phase.

Referring the upstream source to the LV side needs no turns-ratio arithmetic
because Z = V^2/S is evaluated directly with the LV voltage: the ratio squared
cancels between V^2 and the MVA base. This is the standard simplification in
IEC 60909-0 for a study confined to one voltage level.
"""

from __future__ import annotations

import cmath
import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from constants import (
    ALPHA_CU,
    ACB_INSTANTANEOUS_CLEARING_S,
    LOCKED_ROTOR_PF,
    SQRT3,
    STAR_DELTA_CURRENT_RATIO,
    STAR_DELTA_IMPEDANCE_RATIO,
    TEMP_COLD_C,
)

INF = math.inf

DOL = "dol"
STAR_DELTA = "star-delta"


# ===========================================================================
# Cables
# ===========================================================================


@dataclass(frozen=True)
class CableType:
    """One cross-section from the cable library."""

    size_mm2: float
    r20_ohm_km: float
    x_ohm_km: float
    iz_base_a: float


class CableLibrary:
    """Copper/XLPE cable data, loaded from data/cables.yaml."""

    def __init__(self, types: dict[float, CableType], meta: dict[str, Any] | None = None):
        self._types = dict(types)
        self.meta = meta or {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CableLibrary":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        types: dict[float, CableType] = {}
        for size, (r20, x, iz) in raw["sizes"].items():
            s = float(size)
            types[s] = CableType(size_mm2=s, r20_ohm_km=float(r20), x_ohm_km=float(x), iz_base_a=float(iz))
        return cls(types, raw.get("meta", {}))

    def get(self, size_mm2: float) -> CableType:
        key = float(size_mm2)
        if key not in self._types:
            raise KeyError(
                f"cross-section {size_mm2} mm2 is not in the cable library; "
                f"available: {sorted(self._types)}"
            )
        return self._types[key]

    def sizes_ascending(self) -> list[float]:
        return sorted(self._types)

    def __iter__(self) -> Iterable[CableType]:
        return (self._types[s] for s in self.sizes_ascending())

    def __len__(self) -> int:
        return len(self._types)


@dataclass(frozen=True)
class Cable:
    """An installed cable run: a cross-section, a length, and a run count."""

    ctype: CableType
    length_m: float
    runs: int = 1

    def __post_init__(self) -> None:
        if self.length_m <= 0:
            raise ValueError("cable length must be positive")
        if self.runs < 1:
            raise ValueError("cable runs must be at least 1")

    @property
    def size_mm2(self) -> float:
        return self.ctype.size_mm2

    def r_ohm(self, temp_c: float = TEMP_COLD_C) -> float:
        """Conductor resistance at a stated temperature.

        IEC 60228 temperature correction:
            R_theta = R_20 * [1 + alpha * (theta - 20)],  alpha_Cu = 0.00393/K

        R_90 / R_20 = 1.2751, i.e. a hot conductor is 27.5 per cent more
        resistive -- which is why the choice of temperature matters more than
        skin effect at these cross-sections.
        """
        r_per_km = self.ctype.r20_ohm_km * (1.0 + ALPHA_CU * (temp_c - TEMP_COLD_C))
        return r_per_km * (self.length_m / 1000.0) / self.runs

    def x_ohm(self) -> float:
        """Reactance. Independent of temperature; parallel runs halve it."""
        return self.ctype.x_ohm_km * (self.length_m / 1000.0) / self.runs

    def z(self, temp_c: float = TEMP_COLD_C) -> complex:
        return complex(self.r_ohm(temp_c), self.x_ohm())

    def describe(self) -> str:
        return f"{self.runs} x {self.size_mm2:g} mm2, {self.length_m:g} m"


# ===========================================================================
# Source and transformer
# ===========================================================================


@dataclass(frozen=True)
class Source:
    """Upstream utility, as a Thevenin equivalent behind the transformer.

    IEC 60909-0: the network feeder impedance follows from its fault level,
        |Z_Q| = c * U_n^2 / S_kQ
    The voltage factor c is applied at the point of calculation rather than
    being folded in here, so this returns the physical impedance only.
    """

    fault_mva: float
    x_over_r: float

    def z(self, lv_v: float) -> complex:
        """Impedance referred to the LV side, complex ohms per phase."""
        z_mag = (lv_v**2) / (self.fault_mva * 1.0e6)
        phi = math.atan(self.x_over_r)
        return complex(z_mag * math.cos(phi), z_mag * math.sin(phi))


@dataclass(frozen=True)
class Transformer:
    """Two-winding distribution transformer.

    R is derived from the load (copper) loss rather than assumed:
        R_pu = P_loss / S_rated          (both in the same units)
        Z_pu = u_k / 100
        X_pu = sqrt(Z_pu^2 - R_pu^2)
    and converted to ohms on the LV base  Z_base = V_LV^2 / S_rated.

    This is the standard method (IEC 60076-1 test quantities; IEEE Std 242
    Ch. 2) and means the X/R ratio is a consequence of the test certificate,
    not a guess. For this unit: R = 1.05 per cent, X = 4.888 per cent,
    X/R = 4.66, which is typical for a 1000 kVA oil-immersed transformer.
    """

    kva: float
    hv_kv: float
    lv_v: float
    uk_pct: float
    load_loss_kw: float
    vector_group: str = "Dyn11"
    no_load_loss_kw: float = 0.0

    def __post_init__(self) -> None:
        if self.r_pct >= self.uk_pct:
            raise ValueError(
                f"load loss {self.load_loss_kw} kW implies R = {self.r_pct:.3f} per cent, "
                f"which is not less than u_k = {self.uk_pct} per cent; check the test data"
            )

    @property
    def s_va(self) -> float:
        return self.kva * 1000.0

    @property
    def z_base_ohm(self) -> float:
        """LV-side base impedance, Z_base = V^2 / S."""
        return (self.lv_v**2) / self.s_va

    @property
    def flc_a(self) -> float:
        """Rated secondary line current, I = S / (sqrt(3) V)."""
        return self.s_va / (SQRT3 * self.lv_v)

    @property
    def r_pct(self) -> float:
        return 100.0 * self.load_loss_kw / self.kva

    @property
    def x_pct(self) -> float:
        return math.sqrt(self.uk_pct**2 - self.r_pct**2)

    @property
    def z(self) -> complex:
        return complex(
            (self.r_pct / 100.0) * self.z_base_ohm,
            (self.x_pct / 100.0) * self.z_base_ohm,
        )

    @property
    def x_over_r(self) -> float:
        return self.x_pct / self.r_pct


@dataclass(frozen=True)
class Bus:
    nominal_v: float
    frequency_hz: float
    rated_short_time_withstand_ka: float


@dataclass(frozen=True)
class StaticLoad:
    """Non-motor bus load, represented as a constant impedance.

    Constant impedance is the correct choice for a sub-second voltage-dip
    study (IEEE Std 399 Ch. 9): a constant-power model would draw more current
    as the voltage falls, which is the behaviour of a regulated drive over
    seconds, not of lighting and HVAC over the duration of a motor start.
    """

    kva: float
    pf: float

    @property
    def p_kw(self) -> float:
        return self.kva * self.pf

    @property
    def q_kvar(self) -> float:
        return self.kva * math.sin(math.acos(self.pf))


# ===========================================================================
# Motors
# ===========================================================================


@dataclass(frozen=True)
class Motor:
    """Three-phase squirrel-cage induction motor with its supply cable."""

    tag: str
    service: str
    kw: float
    voltage_v: float
    efficiency: float
    pf: float
    lrc_multiple: float
    start_method: str
    t_start_s: float
    t_stall_hot_s: float
    t_stall_cold_s: float
    trip_class: int
    ol_pickup_multiple: float
    cable: Cable
    t_star_s: float | None = None

    def __post_init__(self) -> None:
        if self.start_method not in (DOL, STAR_DELTA):
            raise ValueError(f"{self.tag}: start_method must be {DOL!r} or {STAR_DELTA!r}")
        if self.start_method == STAR_DELTA:
            if self.t_star_s is None:
                raise ValueError(f"{self.tag}: star-delta start needs t_star_s")
            if not 0.0 < self.t_star_s < self.t_start_s:
                raise ValueError(
                    f"{self.tag}: t_star_s ({self.t_star_s} s) must be greater than zero "
                    f"and less than t_start_s ({self.t_start_s} s)"
                )
        if self.t_stall_cold_s < self.t_stall_hot_s:
            raise ValueError(
                f"{self.tag}: cold stall withstand must not be shorter than the hot value"
            )
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError(f"{self.tag}: efficiency out of range")
        if not 0.0 < self.pf <= 1.0:
            raise ValueError(f"{self.tag}: power factor out of range")

    # -- ratings -----------------------------------------------------------

    @property
    def flc_a(self) -> float:
        """Full-load line current.

            I_FLC = P_shaft / (sqrt(3) * V_LL * efficiency * pf)

        The efficiency and power factor convert the SHAFT output rating
        (IEC 60034-1) into the electrical input the cable and protection
        actually see. Computed rather than tabulated, so it can be checked
        against the nameplate as a validation step.
        """
        return (self.kw * 1000.0) / (SQRT3 * self.voltage_v * self.efficiency * self.pf)

    @property
    def s_rated_kva(self) -> float:
        """Apparent power drawn at rated load, S = P_shaft / (eta * pf)."""
        return self.kw / (self.efficiency * self.pf)

    @property
    def p_input_kw(self) -> float:
        return self.kw / self.efficiency

    @property
    def q_input_kvar(self) -> float:
        return self.p_input_kw * math.tan(math.acos(self.pf))

    @property
    def i_lr_a(self) -> float:
        """Nameplate locked-rotor current at rated voltage, I_LR = k * I_FLC."""
        return self.flc_a * self.lrc_multiple

    # -- starting method ---------------------------------------------------

    @property
    def is_star_delta(self) -> bool:
        return self.start_method == STAR_DELTA

    @property
    def t_delta_s(self) -> float:
        """Duration of the delta stage. Equals the whole start for a DOL motor."""
        if not self.is_star_delta:
            return self.t_start_s
        assert self.t_star_s is not None
        return self.t_start_s - self.t_star_s

    @property
    def i_start_initial_a(self) -> float:
        """Line current at the instant of energisation, nameplate basis.

        Star-delta draws one third of the DOL locked-rotor line current --
        see constants.STAR_DELTA_CURRENT_RATIO for the derivation.
        """
        if self.is_star_delta:
            return self.i_lr_a * STAR_DELTA_CURRENT_RATIO
        return self.i_lr_a

    def z_lr(self, connection: str = "delta") -> complex:
        """Locked-rotor impedance, complex ohms per phase.

            |Z_LR| = V_LL / (sqrt(3) * I_LR)      at angle arccos(pf_LR)

        Valid because at standstill (slip = 1) the machine reduces to a fixed
        series impedance -- the constant-impedance representation used for
        motor-starting studies in IEEE Std 399 Ch. 9. pf_LR = 0.25 is the
        adopted value (IEEE Std 242 gives 0.20-0.30 for LV cage machines).

        connection='star' returns the equivalent seen from the line during the
        star stage of a star-delta start, which is 3 x the delta value.
        """
        z_mag = self.voltage_v / (SQRT3 * self.i_lr_a)
        phi = math.acos(LOCKED_ROTOR_PF)
        z = complex(z_mag * math.cos(phi), z_mag * math.sin(phi))
        if connection == "star":
            return z * STAR_DELTA_IMPEDANCE_RATIO
        if connection == "delta":
            return z
        raise ValueError("connection must be 'star' or 'delta'")

    def starting_stages(self) -> list[tuple[float, float, str]]:
        """Current profile of a healthy start on a NAMEPLATE basis.

        Returns [(current_a, duration_s, label), ...]. Nameplate basis means
        rated voltage at the terminals; the achieved current with the real
        supply impedance is lower and is computed in starting.py. Using the
        nameplate value here is the conservative choice for protection
        settings, because it is the largest current the relay could see.
        """
        if self.is_star_delta:
            assert self.t_star_s is not None
            return [
                (self.i_lr_a * STAR_DELTA_CURRENT_RATIO, self.t_star_s, "Star stage"),
                (self.i_lr_a, self.t_delta_s, "Delta stage"),
            ]
        return [(self.i_lr_a, self.t_start_s, "DOL locked-rotor")]

    def time_above(self, threshold_a: float) -> float:
        """Total dwell time above a current threshold during a healthy start.

        This is what sets the locked-rotor timer, and it is where the starting
        method earns its keep: on a star-delta start the star-stage current
        (I_LR/3) sits BELOW the 51LR pickup, so the stall timer only arms for
        the delta stage. The coordination window is correspondingly wider.
        """
        return sum(dur for cur, dur, _ in self.starting_stages() if cur >= threshold_a)


# ===========================================================================
# Feeder breaker
# ===========================================================================


@dataclass(frozen=True)
class FeederBreaker:
    """Incoming ACB with an electronic trip unit, per IEC 60947-2.

    Three elements, the fastest of which wins at any given current:
      long time    I^2*t band above I_r, specified by the tripping time t_r
                   at 6 x I_r, which is the IEC 60947-2 convention
      short time   definite time t_sd above I_sd
      instantaneous  definite time above I_i; None means the element is OFF
    """

    designation: str
    frame_a: float
    icu_ka: float
    ir_a: float
    tr_s: float
    isd_a: float
    tsd_s: float
    ii_a: float | None = None

    def long_time_trip_s(self, i_a: float) -> float:
        """I^2*t long-time band: t = t_r * (6 I_r / I)^2 for I > I_r."""
        if i_a <= self.ir_a:
            return INF
        return self.tr_s * (6.0 * self.ir_a / i_a) ** 2

    def short_time_trip_s(self, i_a: float) -> float:
        return self.tsd_s if i_a >= self.isd_a else INF

    def instantaneous_trip_s(self, i_a: float) -> float:
        if self.ii_a is None:
            return INF
        return ACB_INSTANTANEOUS_CLEARING_S if i_a >= self.ii_a else INF

    def trip_time_s(self, i_a: float) -> float:
        return min(
            self.long_time_trip_s(i_a),
            self.short_time_trip_s(i_a),
            self.instantaneous_trip_s(i_a),
        )


# ===========================================================================
# Plant
# ===========================================================================


@dataclass
class Plant:
    """The whole 415 V bus and everything on it."""

    meta: dict[str, Any]
    source: Source
    transformer: Transformer
    bus: Bus
    static_load: StaticLoad
    motor_diversity_factor: float
    motors: list[Motor]
    feeder: FeederBreaker
    criteria: dict[str, float]
    mccb_frames: list[dict[str, float]]
    cable_library: CableLibrary
    scenario: str | None = None
    scenario_description: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        plant_path: str | Path,
        cables_path: str | Path,
        scenario: str | None = None,
    ) -> "Plant":
        raw = yaml.safe_load(Path(plant_path).read_text(encoding="utf-8"))
        lib = CableLibrary.from_yaml(cables_path)

        description = ""
        if scenario is not None:
            scenarios = raw.get("scenarios") or {}
            if scenario not in scenarios:
                raise KeyError(
                    f"scenario {scenario!r} not defined; available: {sorted(scenarios)}"
                )
            spec = scenarios[scenario]
            description = " ".join((spec.get("description") or "").split())
            raw = _apply_overrides(raw, spec.get("overrides") or {})

        tx_raw = raw["transformer"]
        transformer = Transformer(
            kva=float(tx_raw["kva"]),
            hv_kv=float(tx_raw["hv_kv"]),
            lv_v=float(tx_raw["lv_v"]),
            uk_pct=float(tx_raw["uk_pct"]),
            load_loss_kw=float(tx_raw["load_loss_kw"]),
            vector_group=str(tx_raw.get("vector_group", "Dyn11")),
            no_load_loss_kw=float(tx_raw.get("no_load_loss_kw", 0.0)),
        )

        motors: list[Motor] = []
        for m in raw["motors"]:
            c = m["cable"]
            motors.append(
                Motor(
                    tag=str(m["tag"]),
                    service=str(m["service"]),
                    kw=float(m["kw"]),
                    voltage_v=float(m["voltage_v"]),
                    efficiency=float(m["efficiency"]),
                    pf=float(m["pf"]),
                    lrc_multiple=float(m["lrc_multiple"]),
                    start_method=str(m["start_method"]).strip().lower(),
                    t_start_s=float(m["t_start_s"]),
                    t_star_s=None if m.get("t_star_s") is None else float(m["t_star_s"]),
                    t_stall_hot_s=float(m["t_stall_hot_s"]),
                    t_stall_cold_s=float(m["t_stall_cold_s"]),
                    trip_class=int(m["trip_class"]),
                    ol_pickup_multiple=float(m["ol_pickup_multiple"]),
                    cable=Cable(
                        ctype=lib.get(float(c["size_mm2"])),
                        length_m=float(c["length_m"]),
                        runs=int(c.get("runs", 1)),
                    ),
                )
            )

        tags = [m.tag for m in motors]
        if len(set(tags)) != len(tags):
            raise ValueError("motor tags must be unique")

        fb = raw["feeder_breaker"]
        feeder = FeederBreaker(
            designation=str(fb.get("designation", "ACB-01")),
            frame_a=float(fb["frame_a"]),
            icu_ka=float(fb["icu_ka"]),
            ir_a=float(fb["ir_a"]),
            tr_s=float(fb["tr_s"]),
            isd_a=float(fb["isd_a"]),
            tsd_s=float(fb["tsd_s"]),
            ii_a=None if fb.get("ii_a") is None else float(fb["ii_a"]),
        )

        bus_raw = raw["bus"]
        sl = raw["static_load"]

        return cls(
            meta=raw.get("meta", {}),
            source=Source(
                fault_mva=float(raw["source"]["fault_mva"]),
                x_over_r=float(raw["source"]["x_over_r"]),
            ),
            transformer=transformer,
            bus=Bus(
                nominal_v=float(bus_raw["nominal_v"]),
                frequency_hz=float(bus_raw["frequency_hz"]),
                rated_short_time_withstand_ka=float(bus_raw["rated_short_time_withstand_ka"]),
            ),
            static_load=StaticLoad(kva=float(sl["kva"]), pf=float(sl["pf"])),
            motor_diversity_factor=float(raw.get("motor_diversity_factor", 1.0)),
            motors=motors,
            feeder=feeder,
            criteria={k: float(v) for k, v in (raw.get("criteria") or {}).items()},
            mccb_frames=list(raw.get("mccb_frames") or []),
            cable_library=lib,
            scenario=scenario,
            scenario_description=description,
            raw=raw,
        )

    # -- lookups -----------------------------------------------------------

    def motor(self, tag: str) -> Motor:
        for m in self.motors:
            if m.tag == tag:
                return m
        raise KeyError(f"no motor tagged {tag!r}")

    @property
    def nominal_v(self) -> float:
        return self.bus.nominal_v

    @property
    def largest_motor(self) -> Motor:
        return max(self.motors, key=lambda m: m.kw)

    # -- impedances --------------------------------------------------------

    @property
    def z_source(self) -> complex:
        """Total upstream impedance at the bus: utility + transformer.

        Series addition of the two complex impedances, both referred to 415 V.
        """
        return self.source.z(self.nominal_v) + self.transformer.z

    def load_impedance(self, exclude_tags: Sequence[str] = ()) -> complex | None:
        """Constant-impedance equivalent of the bus load, excluding some motors.

        For a three-phase load drawing S = P + jQ at V_LL, the star-equivalent
        per-phase impedance follows from S_3ph = V_LL^2 / conj(Z):

            |Z| = V_LL^2 / |S|       at angle  +arccos(pf)

        Motors are included at their rated input power scaled by the diversity
        factor; the static load is included in full.
        """
        p_kw = self.static_load.p_kw
        q_kvar = self.static_load.q_kvar
        d = self.motor_diversity_factor
        for m in self.motors:
            if m.tag in exclude_tags:
                continue
            p_kw += d * m.p_input_kw
            q_kvar += d * m.q_input_kvar
        s_va = math.hypot(p_kw, q_kvar) * 1000.0
        if s_va <= 0.0:
            return None
        z_mag = (self.nominal_v**2) / s_va
        phi = math.atan2(q_kvar, p_kw)
        return complex(z_mag * math.cos(phi), z_mag * math.sin(phi))

    # -- loading -----------------------------------------------------------

    @property
    def motor_group_kva(self) -> float:
        return sum(m.s_rated_kva for m in self.motors)

    @property
    def connected_kva(self) -> float:
        """Total connected apparent power, no diversity applied."""
        return self.motor_group_kva + self.static_load.kva

    @property
    def diversified_kva(self) -> float:
        """Apparent power used for the transformer loading check.

        Complex addition, not arithmetic addition of the kVA figures, so the
        differing power factors of the motor group and the static load are
        respected.
        """
        d = self.motor_diversity_factor
        p_kw = self.static_load.p_kw + d * sum(m.p_input_kw for m in self.motors)
        q_kvar = self.static_load.q_kvar + d * sum(m.q_input_kvar for m in self.motors)
        return math.hypot(p_kw, q_kvar)

    @property
    def transformer_loading_pct(self) -> float:
        return 100.0 * self.diversified_kva / self.transformer.kva

    @property
    def diversified_pf(self) -> float:
        d = self.motor_diversity_factor
        p_kw = self.static_load.p_kw + d * sum(m.p_input_kw for m in self.motors)
        s = self.diversified_kva
        return p_kw / s if s > 0 else 1.0


# ===========================================================================
# Scenario overrides
# ===========================================================================


def _apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply dotted-path overrides to the raw YAML tree.

    'transformer.kva' addresses a nested mapping. 'motors.M1.start_method'
    is special-cased: the motor list is keyed by tag rather than by index, so
    a scenario stays readable and does not break when motors are reordered.
    """
    out = copy.deepcopy(raw)
    for path, value in overrides.items():
        parts = path.split(".")
        if parts[0] == "motors":
            if len(parts) < 3:
                raise ValueError(f"motor override {path!r} must be motors.<TAG>.<field>")
            tag = parts[1]
            target = next((m for m in out["motors"] if str(m["tag"]) == tag), None)
            if target is None:
                raise KeyError(f"override {path!r} refers to unknown motor tag {tag!r}")
            node: Any = target
            keys = parts[2:]
        else:
            node = out
            keys = parts
        for k in keys[:-1]:
            if k not in node:
                raise KeyError(f"override path {path!r} does not exist in the plant data")
            node = node[k]
        leaf = keys[-1]
        if leaf not in node:
            raise KeyError(f"override path {path!r} does not exist in the plant data")
        node[leaf] = value
    return out


# ===========================================================================
# Small complex helpers
# ===========================================================================


def parallel(*impedances: complex) -> complex:
    """Parallel combination by admittance summation.

    Used for the motor short-circuit contribution and for the running bus load
    during a motor start. Admittance summation keeps the phase angles right,
    which arithmetic addition of magnitudes would not: the network X/R is
    about 5 whereas an LV motor is 1/0.42 = 2.4, so the two contributions do
    not add in phase.
    """
    y = 0.0 + 0.0j
    for z in impedances:
        if z == 0:
            return 0.0 + 0.0j
        y += 1.0 / z
    if y == 0:
        return complex(INF, 0.0)
    return 1.0 / y


def polar_str(z: complex, unit: str = "ohm") -> str:
    mag, ang = cmath.polar(z)
    return f"{mag:.6f} {unit} at {math.degrees(ang):.1f} deg"
