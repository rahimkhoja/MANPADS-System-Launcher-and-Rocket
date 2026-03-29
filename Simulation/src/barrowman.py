"""
Barrowman Aerodynamics Calculator
=================================
Subsonic aerodynamic prediction for fin-stabilised rockets using the
Barrowman method (1966) with extensions for canards and drag estimation.

Reference:
  J.S. Barrowman, "The Theoretical Prediction of the Center of Pressure",
  SRAD, 1966.
"""

import math
from dataclasses import dataclass, field
from typing import Tuple, List


# ── Geometry Inputs ──────────────────────────────────────────────────

@dataclass
class NoseCone:
    shape: str = "ogive"   # "conical", "ogive", "vonkarman", "parabolic"
    length: float = 0.038  # m
    base_radius: float = 0.0375  # m


@dataclass
class BodyTube:
    length: float = 0.549  # m (total body minus nose)
    radius: float = 0.0375  # m
    boat_tail: bool = False
    boat_tail_length: float = 0.03
    boat_tail_end_radius: float = 0.030


@dataclass
class FinSet:
    count: int = 4
    root_chord: float = 0.040   # m
    tip_chord: float = 0.020    # m
    span: float = 0.025         # m (semi-span beyond body)
    sweep_angle: float = 45.0   # deg (leading-edge sweep)
    thickness: float = 0.002    # m
    position_from_nose: float = 0.50  # m (leading-edge root position)


@dataclass
class CanardSet:
    count: int = 4
    root_chord: float = 0.030   # m
    tip_chord: float = 0.015    # m
    span: float = 0.015         # m (semi-span beyond body)
    sweep_angle: float = 0.0    # deg
    thickness: float = 0.001    # m
    position_from_nose: float = 0.08  # m


@dataclass
class RocketGeometry:
    nose: NoseCone = field(default_factory=NoseCone)
    body: BodyTube = field(default_factory=BodyTube)
    fins: FinSet = field(default_factory=FinSet)
    canards: CanardSet = field(default_factory=CanardSet)

    @property
    def total_length(self) -> float:
        return self.nose.length + self.body.length

    @property
    def diameter(self) -> float:
        return 2 * self.body.radius

    @property
    def reference_area(self) -> float:
        return math.pi * self.body.radius ** 2


# ── Barrowman Results ────────────────────────────────────────────────

@dataclass
class AeroResult:
    CNa_nose: float = 0.0     # per radian
    CNa_fins: float = 0.0
    CNa_canards: float = 0.0
    CNa_total: float = 0.0

    CP_nose: float = 0.0      # m from nose tip
    CP_fins: float = 0.0
    CP_canards: float = 0.0
    CP_total: float = 0.0     # combined CP

    Cd0: float = 0.0          # zero-lift drag coefficient
    Cd_base: float = 0.0
    Cd_skin: float = 0.0
    Cd_fin_pressure: float = 0.0
    Cd_nose_pressure: float = 0.0

    stability_cal: float = 0.0   # stability margin in calibres
    roll_authority: float = 0.0  # Nm per radian of canard deflection at q_bar=1

    def summary(self) -> str:
        lines = [
            "=== Barrowman Aero Results ===",
            f"CNa total:   {self.CNa_total:.3f} /rad",
            f"  nose:      {self.CNa_nose:.3f}",
            f"  fins:      {self.CNa_fins:.3f}",
            f"  canards:   {self.CNa_canards:.3f}",
            f"CP total:    {self.CP_total*1000:.1f} mm from nose",
            f"  nose:      {self.CP_nose*1000:.1f} mm",
            f"  fins:      {self.CP_fins*1000:.1f} mm",
            f"  canards:   {self.CP_canards*1000:.1f} mm",
            f"Stability:   {self.stability_cal:.2f} calibres",
            f"Cd0:         {self.Cd0:.4f}",
            f"  skin:      {self.Cd_skin:.4f}",
            f"  base:      {self.Cd_base:.4f}",
            f"  fin press: {self.Cd_fin_pressure:.4f}",
            f"  nose:      {self.Cd_nose_pressure:.4f}",
            f"Roll auth:   {self.roll_authority:.6f} Nm/(rad·Pa)",
        ]
        return "\n".join(lines)


# ── Barrowman Calculator ─────────────────────────────────────────────

def _nose_CNa_and_CP(nose: NoseCone) -> Tuple[float, float]:
    """Normal-force slope and CP location for the nose cone."""
    CNa = 2.0  # all axisymmetric nose shapes give CNa = 2 per radian

    if nose.shape == "conical":
        CP = 2.0 / 3.0 * nose.length
    elif nose.shape == "ogive":
        CP = 0.466 * nose.length
    elif nose.shape == "vonkarman":
        CP = 0.50 * nose.length
    elif nose.shape == "parabolic":
        CP = 0.50 * nose.length
    else:
        CP = 0.466 * nose.length  # default ogive

    return CNa, CP


def _finset_CNa_and_CP(fins: FinSet, body_radius: float,
                       nose_length: float) -> Tuple[float, float]:
    """Barrowman normal-force slope and CP for a set of trapezoidal fins."""
    N = fins.count
    s = fins.span
    Cr = fins.root_chord
    Ct = fins.tip_chord
    d = 2 * body_radius
    r = body_radius
    sweep_rad = math.radians(fins.sweep_angle)

    mid_chord_sweep_dist = s * math.tan(sweep_rad) + 0.5 * Ct - 0.5 * Cr
    l_m = math.sqrt(s ** 2 + mid_chord_sweep_dist ** 2) if s > 0 else Cr

    planform_area = 0.5 * (Cr + Ct) * s
    AR = (2 * s ** 2) / planform_area if planform_area > 1e-9 else 0.0

    if AR < 1e-6:
        return 0.0, fins.position_from_nose + 0.5 * Cr

    # Barrowman fin CNa (Barrowman 1966, eq. 3.38)
    # CNa = K_fb * 4N(s/d)^2 / (1 + sqrt(1 + (AR/(2*cos(sweep_mid)))^2))
    K_fb = 1 + r / (r + s)  # fin-body interference factor
    cos_sweep = max(abs(math.cos(sweep_rad)), 0.01)
    half_AR_cos = AR / (2 * cos_sweep)
    denom = 1 + math.sqrt(1 + half_AR_cos ** 2)
    CNa_one_pair = (4 * N * (s / d) ** 2) / denom
    CNa = K_fb * CNa_one_pair

    # CP of fin set (Barrowman eq. for trapezoidal fins)
    x_mac = (Cr - Ct) / 3.0 * (Cr + 2 * Ct) / (Cr + Ct) if (Cr + Ct) > 1e-9 else 0.0
    sweep_dist = s * math.tan(sweep_rad)
    CP_on_fin = x_mac + sweep_dist / 3.0 * (Cr + 2 * Ct) / (Cr + Ct) if (Cr + Ct) > 1e-9 else 0.0
    CP = fins.position_from_nose + CP_on_fin

    return CNa, CP


def compute_barrowman(geom: RocketGeometry,
                      cg_from_nose: float = 0.35,
                      mach: float = 0.0,
                      Re: float = 1e6) -> AeroResult:
    """Full Barrowman analysis for a canard + fin rocket.

    Parameters
    ----------
    geom : RocketGeometry
    cg_from_nose : float, m
    mach : float, freestream Mach (for compressibility correction)
    Re : float, Reynolds number (for skin friction)

    Returns
    -------
    AeroResult with all coefficients and locations
    """
    res = AeroResult()
    r = geom.body.radius
    d = geom.diameter
    S_ref = geom.reference_area

    # Prandtl-Glauert compressibility correction
    beta_pg = math.sqrt(max(1 - mach ** 2, 0.1)) if mach < 0.9 else 1.0

    # ── Normal-force slopes ──
    res.CNa_nose, res.CP_nose = _nose_CNa_and_CP(geom.nose)

    res.CNa_fins, res.CP_fins = _finset_CNa_and_CP(
        geom.fins, r, geom.nose.length)

    res.CNa_canards, res.CP_canards = _finset_CNa_and_CP(
        geom.canards, r, geom.nose.length)

    # Apply compressibility to fin/canard slopes
    res.CNa_fins /= beta_pg
    res.CNa_canards /= beta_pg

    res.CNa_total = res.CNa_nose + res.CNa_fins + res.CNa_canards

    # ── Combined CP (weighted average) ──
    if res.CNa_total > 1e-9:
        res.CP_total = (res.CNa_nose * res.CP_nose
                        + res.CNa_fins * res.CP_fins
                        + res.CNa_canards * res.CP_canards) / res.CNa_total
    else:
        res.CP_total = 0.5 * geom.total_length

    # ── Stability margin ──
    res.stability_cal = (res.CP_total - cg_from_nose) / d

    # ── Drag estimation ──

    # Skin friction (turbulent flat-plate, Schlichting)
    Cf = 0.455 / (max(math.log10(max(Re, 1e3)), 1.0) ** 2.58)

    wetted_body = 2 * math.pi * r * geom.body.length
    wetted_nose = math.pi * r * math.sqrt(r ** 2 + geom.nose.length ** 2)
    fineness = geom.body.length / d if d > 0 else 10.0
    body_form_factor = 1 + 60 / fineness ** 3 + 0.0025 * fineness

    res.Cd_skin = Cf * body_form_factor * (wetted_body + wetted_nose) / S_ref

    # Add fin skin friction
    for fset in [geom.fins, geom.canards]:
        S_fin_wet = 2 * fset.count * 0.5 * (fset.root_chord + fset.tip_chord) * fset.span
        tc = fset.thickness / max(0.5 * (fset.root_chord + fset.tip_chord), 0.001)
        fin_form = 1 + 2 * tc
        res.Cd_skin += Cf * fin_form * S_fin_wet / S_ref

    # Base drag
    res.Cd_base = 0.029 / math.sqrt(max(Cf, 1e-6))
    if geom.body.boat_tail:
        taper_ratio = geom.body.boat_tail_end_radius / r if r > 0 else 1.0
        res.Cd_base *= taper_ratio ** 2

    # Nose pressure drag
    if geom.nose.shape == "conical":
        half_angle = math.atan2(r, geom.nose.length) if geom.nose.length > 0 else math.pi / 6
        res.Cd_nose_pressure = 2 * math.sin(half_angle) ** 2
    elif geom.nose.shape == "ogive":
        res.Cd_nose_pressure = 0.1 * (r / max(geom.nose.length, 0.001)) ** 2
    elif geom.nose.shape == "vonkarman":
        res.Cd_nose_pressure = 0.05 * (r / max(geom.nose.length, 0.001)) ** 2
    else:
        res.Cd_nose_pressure = 0.15 * (r / max(geom.nose.length, 0.001)) ** 2

    # Fin pressure drag (LE bluntness)
    for fset in [geom.fins, geom.canards]:
        S_proj = fset.count * fset.thickness * fset.span
        res.Cd_fin_pressure += 1.2 * S_proj / S_ref

    res.Cd0 = res.Cd_skin + res.Cd_base + res.Cd_nose_pressure + res.Cd_fin_pressure

    # ── Roll authority ──
    # Torque per radian of canard deflection per unit of dynamic pressure
    c = geom.canards
    canard_planform = 0.5 * (c.root_chord + c.tip_chord) * c.span
    canard_Cla = 2 * math.pi  # thin airfoil theory
    AR_c = (2 * c.span ** 2) / canard_planform if canard_planform > 1e-9 else 2.0
    canard_Cla_3d = canard_Cla / (1 + canard_Cla / (math.pi * AR_c)) if AR_c > 0 else canard_Cla
    arm = r + 0.5 * c.span
    res.roll_authority = c.count * canard_planform * canard_Cla_3d * arm

    return res


# ── Convenience: build geometry from RocketParameters ────────────────

def geometry_from_params(params) -> RocketGeometry:
    """Build a RocketGeometry from a rocket_dynamics.RocketParameters instance.

    Uses the improved design geometry (Von Karman nose, large folding fins,
    boat tail) matching the design_sweep.py winning configuration.
    """
    r = params.diameter / 2
    nose_len = getattr(params, 'nose_length', 0.120)
    body_len = getattr(params, 'body_length', 0.549)
    return RocketGeometry(
        nose=NoseCone(
            shape="vonkarman",
            length=nose_len,
            base_radius=r,
        ),
        body=BodyTube(
            length=body_len,
            radius=r,
            boat_tail=True,
            boat_tail_length=0.03,
            boat_tail_end_radius=0.030,
        ),
        fins=FinSet(
            count=getattr(params, 'num_fins', 4),
            root_chord=0.070,
            tip_chord=0.035,
            span=0.060,
            sweep_angle=30.0,
            thickness=0.002,
            position_from_nose=body_len + nose_len - 0.070 - 0.01,
        ),
        canards=CanardSet(
            count=params.num_canards,
            root_chord=0.045,
            tip_chord=0.0225,
            span=0.020,
            sweep_angle=0.0,
            thickness=0.001,
            position_from_nose=getattr(params, 'canard_from_nose', 0.10),
        ),
    )


# ── CLI quick-test ───────────────────────────────────────────────────

if __name__ == "__main__":
    geom = RocketGeometry()
    result = compute_barrowman(geom, cg_from_nose=0.35)
    print(result.summary())
