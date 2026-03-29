#!/usr/bin/env python3
"""
Headless rocket previews: shaded matplotlib views (no GPU window).

- Single STL: smooth Lambert shading on one material.
- --preset improved-deployed: parametric winner geometry with fins & canards
  fully extended (same as sweep STLs), body / fins / canards colored separately.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import trimesh

# Parametric generator (same tree as this script)
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC = _SCRIPT_DIR.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(str(path), force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def subsample_faces(faces: np.ndarray, max_faces: int) -> np.ndarray:
    if len(faces) <= max_faces:
        return faces
    idx = np.linspace(0, len(faces) - 1, max_faces, dtype=int)
    return faces[idx]


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln = np.maximum(ln, 1e-12)
    return n / ln


def lambert_face_rgba(
    vertices: np.ndarray,
    faces: np.ndarray,
    base_rgb: tuple[float, float, float],
    light_dir: np.ndarray,
    ambient: float = 0.18,
    diffuse: float = 0.82,
    alpha: float = 1.0,
) -> np.ndarray:
    """Per-face RGBA from a single directional light (camera-independent)."""
    L = light_dir / max(np.linalg.norm(light_dir), 1e-12)
    n = _face_normals(vertices, faces)
    d = np.clip(np.einsum("ij,j->i", n, L), 0.0, 1.0)
    s = ambient + diffuse * d
    r, g, b = base_rgb
    rgba = np.column_stack([r * s, g * s, b * s, np.full(len(faces), alpha)])
    return np.clip(rgba, 0.0, 1.0)


def mesh_to_collection(
    mesh: trimesh.Trimesh,
    base_rgb: tuple[float, float, float],
    light_dir: np.ndarray,
    max_faces: int,
    edge_rgba: tuple[float, float, float, float] = (0.06, 0.08, 0.12, 0.35),
) -> Poly3DCollection:
    faces = subsample_faces(mesh.faces, max_faces)
    verts = mesh.vertices
    polys = verts[faces]
    fc = lambert_face_rgba(verts, faces, base_rgb, light_dir)
    coll = Poly3DCollection(polys, linewidths=0.06, antialiased=True)
    coll.set_facecolor(fc)
    coll.set_edgecolor(edge_rgba)
    return coll


def add_mesh_to_ax(ax, mesh: trimesh.Trimesh, **kwargs) -> None:
    coll = mesh_to_collection(mesh, **kwargs)
    ax.add_collection3d(coll)


def autoscale_ax(ax, vertices: np.ndarray, margin: float = 1.12) -> None:
    lim = np.abs(vertices).max() * margin
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))


def render_triptych(
    meshes_colors: list[tuple[trimesh.Trimesh, tuple[float, float, float]]],
    out_png: Path,
    title: str,
    max_faces: int = 16000,
    scale_mm: bool = True,
    dpi: int = 200,
) -> None:
    """Render several meshes (same frame) with distinct base colors."""
    combined = trimesh.util.concatenate([m.copy() for m, _ in meshes_colors])
    if scale_mm:
        for m, _ in meshes_colors:
            m.apply_scale(1000.0)
        combined.apply_scale(1000.0)

    light = np.array([0.55, 0.35, 0.85], dtype=float)

    views = [
        ("Side (XZ)", 12, -72, 0),
        ("Top (XY)", 89, -90, 0),
        ("Three-quarter", 22, -58, 0),
    ]

    fig = plt.figure(figsize=(14.5, 4.8), dpi=dpi, facecolor="#14161c")
    fig.suptitle(title, fontsize=12, color="#e8eaef", y=0.98)

    all_verts = []
    for i, (name, elev, azim, roll) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.set_facecolor("#14161c")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#2a2f3a")
        ax.yaxis.pane.set_edgecolor("#2a2f3a")
        ax.zaxis.pane.set_edgecolor("#2a2f3a")
        ax.tick_params(colors="#8b92a5", labelsize=7)
        ax.xaxis.label.set_color("#a8b0c4")
        ax.yaxis.label.set_color("#a8b0c4")
        ax.zaxis.label.set_color("#a8b0c4")

        for mesh, rgb in meshes_colors:
            add_mesh_to_ax(ax, mesh, base_rgb=rgb, light_dir=light, max_faces=max_faces)

        v = combined.vertices
        all_verts.append(v)
        autoscale_ax(ax, v, 1.14)
        ax.set_xlabel("X (mm)" if scale_mm else "X (m)")
        ax.set_ylabel("Y (mm)" if scale_mm else "Y (m)")
        ax.set_zlabel("Z (mm)" if scale_mm else "Z (m)")
        ax.set_title(name, fontsize=10, color="#c5cad8", pad=8)
        ax.view_init(elev=elev, azim=azim, roll=roll)
        ax.grid(False)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def improved_deployed_meshes():
    """Barrowman sweep winner #1 style: 120 mm VK nose, 60 mm fins, 20 mm canards, boat tail."""
    from rocket_geometry import generate_rocket

    BODY_LENGTH = 0.549
    NOSE_LEN = 0.120
    fin_root = 0.070
    fin_pos = max(0.02, BODY_LENGTH - fin_root - 0.01)
    total_len = NOSE_LEN + BODY_LENGTH
    canard_z = total_len - 0.10  # 100 mm from nose tip

    parts = generate_rocket(
        nose_shape="vonkarman",
        nose_length=NOSE_LEN,
        body_length=BODY_LENGTH,
        body_radius=0.0375,
        boat_tail=True,
        fin_count=4,
        fin_root=fin_root,
        fin_tip=fin_root * 0.5,
        fin_span=0.060,
        fin_sweep=30.0,
        fin_thickness=0.002,
        fin_position=fin_pos,
        canard_count=4,
        canard_root=0.045,
        canard_tip=0.045 * 0.5,
        canard_span=0.020,
        canard_sweep=0.0,
        canard_thickness=0.001,
        canard_position=canard_z,
        return_parts=True,
    )

    nose, body, fins, canards = parts[0], parts[1], parts[2], parts[3]
    return [
        (nose, (0.52, 0.56, 0.62)),   # nose — light alloy
        (body, (0.42, 0.46, 0.52)),   # body — darker tube
        (fins, (0.78, 0.38, 0.22)),    # tail fins — copper
        (canards, (0.92, 0.72, 0.28)), # canards — gold
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="Render shaded STL / preset rocket previews")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--stl", help="Input STL path")
    g.add_argument("--preset", choices=["improved-deployed"], help="Parametric geometry preset")

    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--title", default="", help="Figure title")
    p.add_argument("--max-faces", type=int, default=16000)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    if args.preset == "improved-deployed":
        meshes_colors = improved_deployed_meshes()
        title = args.title or "Improved design — fins & canards deployed (parametric)"
        render_triptych(
            meshes_colors,
            Path(args.out),
            title,
            max_faces=args.max_faces,
            dpi=args.dpi,
        )
        print(f"Wrote {args.out}")
        return 0

    stl = Path(args.stl)
    if not stl.is_file():
        print(f"Missing STL: {stl}", file=sys.stderr)
        return 1

    mesh = load_mesh(stl)
    title = args.title or stl.stem.replace("_", " ")
    rgb = (0.48, 0.62, 0.88)
    render_triptych([(mesh, rgb)], Path(args.out), title, max_faces=args.max_faces, dpi=args.dpi)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
