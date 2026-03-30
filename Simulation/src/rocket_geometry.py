"""
Parametric Rocket STL Generator
================================
Creates watertight STL meshes from geometric parameters using trimesh.
Supports conical, ogive, and Von Karman nose cones, cylindrical body tubes
with optional boat tail, trapezoidal fins, and flat-plate canards.
"""

import math
import numpy as np

try:
    import trimesh
    from trimesh.creation import cylinder, extrude_polygon
except ImportError:
    trimesh = None


# ── Nose Cone Generators ─────────────────────────────────────────────

def _profile_conical(t: np.ndarray, radius: float) -> np.ndarray:
    return radius * t


def _profile_ogive(t: np.ndarray, radius: float, length: float) -> np.ndarray:
    rho = (radius ** 2 + length ** 2) / (2 * radius)
    x = t * length
    return np.sqrt(rho ** 2 - (length - x) ** 2) + radius - rho


def _profile_vonkarman(t: np.ndarray, radius: float) -> np.ndarray:
    theta = np.arccos(1 - 2 * t)
    return radius / math.sqrt(math.pi) * np.sqrt(theta - np.sin(2 * theta) / 2)


def make_nose_cone(shape: str, length: float, radius: float,
                   n_axial: int = 40, n_circ: int = 32) -> "trimesh.Trimesh":
    """Generate a nose cone mesh.

    Parameters
    ----------
    shape : str
        One of "conical", "ogive", "vonkarman".
    length : float
        Nose length in metres.
    radius : float
        Base radius in metres.
    n_axial : int
        Number of slices along length.
    n_circ : int
        Number of circumferential segments.
    """
    if trimesh is None:
        raise ImportError("trimesh required for geometry generation")

    t = np.linspace(0, 1, n_axial)
    if shape == "conical":
        r = _profile_conical(t, radius)
    elif shape == "ogive":
        r = _profile_ogive(t, radius, length)
    elif shape == "vonkarman":
        r = _profile_vonkarman(t, radius)
    else:
        r = _profile_ogive(t, radius, length)

    r[0] = 0.0001  # avoid degenerate tip
    z = t * length

    theta = np.linspace(0, 2 * math.pi, n_circ, endpoint=False)

    verts = []
    # Tip vertex
    verts.append([0, 0, length])
    for i in range(n_axial):
        for j in range(n_circ):
            x = r[i] * np.cos(theta[j])
            y = r[i] * np.sin(theta[j])
            verts.append([x, y, length - z[i]])

    verts = np.array(verts)
    faces = []

    # Tip fan
    for j in range(n_circ):
        j_next = (j + 1) % n_circ
        faces.append([0, 1 + j, 1 + j_next])

    # Body quads
    for i in range(n_axial - 1):
        for j in range(n_circ):
            j_next = (j + 1) % n_circ
            a = 1 + i * n_circ + j
            b = 1 + i * n_circ + j_next
            c = 1 + (i + 1) * n_circ + j_next
            d = 1 + (i + 1) * n_circ + j
            faces.append([a, d, c])
            faces.append([a, c, b])

    # Watertight base cap (disk at z=0) — snappyHexMesh needs closed surface
    base_start = 1 + (n_axial - 1) * n_circ
    verts_list = verts.tolist()
    center_idx = len(verts_list)
    verts_list.append([0.0, 0.0, 0.0])
    verts = np.array(verts_list)
    cap_faces = []
    for j in range(n_circ):
        j_next = (j + 1) % n_circ
        b = base_start + j
        c = base_start + j_next
        cap_faces.append([center_idx, c, b])
    faces = np.vstack([np.array(faces), np.array(cap_faces)])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    mesh.fix_normals()
    return mesh


# ── Body Tube ────────────────────────────────────────────────────────

def make_body_tube(length: float, radius: float,
                   boat_tail: bool = False,
                   bt_length: float = 0.03,
                   bt_end_radius: float = 0.030,
                   n_circ: int = 32) -> "trimesh.Trimesh":
    """Cylindrical body tube, optionally with an aft boat tail."""
    if trimesh is None:
        raise ImportError("trimesh required")

    if boat_tail:
        main_len = length - bt_length
        main = cylinder(radius=radius, height=main_len, sections=n_circ)
        main.apply_translation([0, 0, main_len / 2])

        n_bt = 10
        z_bt = np.linspace(0, bt_length, n_bt)
        r_bt = np.linspace(radius, bt_end_radius, n_bt)
        theta = np.linspace(0, 2 * math.pi, n_circ, endpoint=False)

        verts = []
        for i in range(n_bt):
            for j in range(n_circ):
                verts.append([r_bt[i] * np.cos(theta[j]),
                              r_bt[i] * np.sin(theta[j]),
                              -z_bt[i]])

        faces = []
        for i in range(n_bt - 1):
            for j in range(n_circ):
                jn = (j + 1) % n_circ
                a = i * n_circ + j
                b = i * n_circ + jn
                c = (i + 1) * n_circ + jn
                d = (i + 1) * n_circ + j
                faces.append([a, d, c])
                faces.append([a, c, b])

        bt_mesh = trimesh.Trimesh(vertices=np.array(verts),
                                  faces=np.array(faces))
        bt_mesh.fix_normals()
        combined = trimesh.util.concatenate([main, bt_mesh])
        return combined
    else:
        mesh = cylinder(radius=radius, height=length, sections=n_circ)
        mesh.apply_translation([0, 0, length / 2])
        return mesh


# ── Fin / Canard ─────────────────────────────────────────────────────

def _make_single_fin(root_chord: float, tip_chord: float, span: float,
                     sweep_angle_deg: float, thickness: float,
                     body_radius: float) -> "trimesh.Trimesh":
    """Single trapezoidal fin as an extruded polygon."""
    sweep = math.tan(math.radians(sweep_angle_deg)) * span

    # 2D profile in the (chord, span) plane
    pts = np.array([
        [0, 0],
        [root_chord, 0],
        [sweep + tip_chord, span],
        [sweep, span],
    ])

    mesh = trimesh.creation.extrude_polygon(
        trimesh.path.polygons.polygon_from_points(pts),
        height=thickness
    ) if hasattr(trimesh.path, 'polygons') else _extrude_quad(pts, thickness)

    mesh.apply_translation([0, 0, -thickness / 2])

    # Rotate so span points radially outward (+Y) and chord along +Z
    R = trimesh.transformations.euler_matrix(math.pi / 2, 0, -math.pi / 2)
    mesh.apply_transform(R)
    mesh.apply_translation([0, body_radius, 0])

    return mesh


def _extrude_quad(pts_2d: np.ndarray, thickness: float) -> "trimesh.Trimesh":
    """Manually extrude a 2D quad into a thin prism."""
    n = len(pts_2d)
    bottom = np.column_stack([pts_2d, np.zeros(n)])
    top = np.column_stack([pts_2d, np.full(n, thickness)])
    verts = np.vstack([bottom, top])

    faces = []
    # Bottom and top faces (fan triangulation)
    for i in range(1, n - 1):
        faces.append([0, i + 1, i])
        faces.append([n, n + i, n + i + 1])
    # Side faces
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    mesh.fix_normals()
    return mesh


def make_fin_set(count: int, root_chord: float, tip_chord: float,
                 span: float, sweep_angle_deg: float, thickness: float,
                 body_radius: float, z_position: float,
                 deflection_deg: float = 0.0) -> "trimesh.Trimesh":
    """Generate a set of fins equally spaced around the body.

    deflection_deg rotates each fin about its hinge axis (tangent to body
    surface at the fin root, i.e. the body-axis Z direction at the fin's
    azimuthal position). Positive deflection tilts the fin tip in the +Z
    (nose) direction. All fins deflect identically.
    """
    meshes = []
    for i in range(count):
        azimuth = 2 * math.pi * i / count
        fin = _make_single_fin(root_chord, tip_chord, span,
                               sweep_angle_deg, thickness, body_radius)
        if abs(deflection_deg) > 0.01:
            hinge_pt = np.array([0.0, body_radius, 0.0])
            hinge_axis = np.array([0.0, 0.0, 1.0])
            T = trimesh.transformations.rotation_matrix(
                math.radians(deflection_deg), hinge_axis, hinge_pt)
            fin.apply_transform(T)

        R = trimesh.transformations.rotation_matrix(azimuth, [0, 0, 1])
        fin.apply_transform(R)
        fin.apply_translation([0, 0, z_position])
        meshes.append(fin)

    return trimesh.util.concatenate(meshes)


# ── Rail Buttons ─────────────────────────────────────────────────────

def make_rail_buttons(body_radius: float, body_length: float,
                      button_radius: float = 0.003,
                      button_height: float = 0.004,
                      n_circ: int = 16) -> "trimesh.Trimesh":
    """Two small cylindrical rail buttons on one side of the body.

    Placed at 25% and 75% of body length, protruding radially outward.
    """
    if trimesh is None:
        raise ImportError("trimesh required")

    positions_z = [body_length * 0.25, body_length * 0.75]
    buttons = []
    for zp in positions_z:
        btn = cylinder(radius=button_radius, height=button_height, sections=n_circ)
        R = trimesh.transformations.euler_matrix(0, math.pi / 2, 0)
        btn.apply_transform(R)
        btn.apply_translation([body_radius + button_height / 2, 0, zp])
        buttons.append(btn)
    return trimesh.util.concatenate(buttons)


# ── Rail Launcher ────────────────────────────────────────────────────

def generate_rail_launcher(rail_length: float = 1.0,
                           rail_spacing: float = 0.060,
                           rail_profile: float = 0.0254,
                           base_width: float = 0.20,
                           base_depth: float = 0.15,
                           base_thickness: float = 0.006,
                           n_circ: int = 16) -> "trimesh.Trimesh":
    """Parametric rail launcher: two parallel rails on a flat base plate.

    Parameters
    ----------
    rail_length : float
        Length of each rail (m).
    rail_spacing : float
        Center-to-center distance between rails (m).
    rail_profile : float
        Cross-section size of each rail (square extrusion, m). Default 1010 = 25.4 mm.
    base_width, base_depth, base_thickness : float
        Dimensions of the rectangular base plate (m).
    """
    if trimesh is None:
        raise ImportError("trimesh required")

    half_s = rail_spacing / 2
    half_p = rail_profile / 2

    parts = []

    for sign in (-1, 1):
        rail = trimesh.creation.box(
            extents=[rail_profile, rail_profile, rail_length]
        )
        rail.apply_translation([sign * half_s, 0, rail_length / 2])
        parts.append(rail)

    base = trimesh.creation.box(
        extents=[base_width, base_depth, base_thickness]
    )
    base.apply_translation([0, 0, -base_thickness / 2])
    parts.append(base)

    igniter_mount = cylinder(radius=0.010, height=0.025, sections=n_circ)
    igniter_mount.apply_translation([0, 0, 0.0125])
    parts.append(igniter_mount)

    launcher = trimesh.util.concatenate(parts)
    launcher.fix_normals()
    return launcher


# ── Full Rocket Assembly ─────────────────────────────────────────────

def generate_rocket(nose_shape: str = "ogive",
                    nose_length: float = 0.038,
                    body_length: float = 0.549,
                    body_radius: float = 0.0375,
                    boat_tail: bool = False,
                    bt_length: float = 0.03,
                    bt_end_radius: float = 0.030,
                    fin_count: int = 4,
                    fin_root: float = 0.040,
                    fin_tip: float = 0.020,
                    fin_span: float = 0.025,
                    fin_sweep: float = 45.0,
                    fin_thickness: float = 0.002,
                    fin_position: float = None,
                    canard_count: int = 4,
                    canard_root: float = 0.030,
                    canard_tip: float = 0.015,
                    canard_span: float = 0.015,
                    canard_sweep: float = 0.0,
                    canard_thickness: float = 0.001,
                    canard_position: float = None,
                    n_circ: int = 32,
                    return_parts: bool = False,
                    rail_buttons: bool = False,
                    canard_deflection_deg: float = 0.0):
    """Generate a complete rocket STL from parametric inputs.

    The rocket is oriented along +Z with nose tip at the top.
    Returns a single concatenated trimesh, or a list of parts if return_parts.

    All dimensions in metres.
    """
    if trimesh is None:
        raise ImportError("trimesh required for geometry generation")

    total_length = nose_length + body_length
    if fin_position is None:
        fin_position = 0.02
    if canard_position is None:
        canard_position = total_length - nose_length - 0.05

    parts = []

    nose = make_nose_cone(nose_shape, nose_length, body_radius, n_circ=n_circ)
    nose.apply_translation([0, 0, body_length])
    parts.append(nose)

    body = make_body_tube(body_length, body_radius, boat_tail,
                          bt_length, bt_end_radius, n_circ=n_circ)
    parts.append(body)

    if fin_count > 0 and fin_span > 0.001:
        fins = make_fin_set(fin_count, fin_root, fin_tip, fin_span,
                            fin_sweep, fin_thickness, body_radius,
                            fin_position)
        parts.append(fins)

    if canard_count > 0 and canard_span > 0.001:
        canards = make_fin_set(canard_count, canard_root, canard_tip,
                               canard_span, canard_sweep, canard_thickness,
                               body_radius, canard_position,
                               deflection_deg=canard_deflection_deg)
        parts.append(canards)

    if rail_buttons:
        btns = make_rail_buttons(body_radius, body_length)
        parts.append(btns)

    if return_parts:
        for p in parts:
            p.fix_normals()
        return parts

    rocket = trimesh.util.concatenate(parts)
    rocket.fix_normals()
    return rocket


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate parametric rocket STL")
    parser.add_argument("--output", default="rocket_parametric.stl")
    parser.add_argument("--nose", default="ogive", choices=["conical", "ogive", "vonkarman"])
    parser.add_argument("--nose-length", type=float, default=0.060)
    parser.add_argument("--body-length", type=float, default=0.549)
    parser.add_argument("--body-radius", type=float, default=0.0375)
    parser.add_argument("--fin-count", type=int, default=4)
    parser.add_argument("--fin-span", type=float, default=0.025)
    parser.add_argument("--canard-count", type=int, default=4)
    parser.add_argument("--canard-span", type=float, default=0.015)
    parser.add_argument("--boat-tail", action="store_true")
    args = parser.parse_args()

    mesh = generate_rocket(
        nose_shape=args.nose,
        nose_length=args.nose_length,
        body_length=args.body_length,
        body_radius=args.body_radius,
        boat_tail=args.boat_tail,
        fin_count=args.fin_count,
        fin_span=args.fin_span,
        canard_count=args.canard_count,
        canard_span=args.canard_span,
    )
    mesh.export(args.output)
    bb = mesh.bounds
    dims = bb[1] - bb[0]
    print(f"Exported {args.output}: {len(mesh.faces)} faces")
    print(f"Dimensions: {dims[0]*1000:.1f} x {dims[1]*1000:.1f} x {dims[2]*1000:.1f} mm")
