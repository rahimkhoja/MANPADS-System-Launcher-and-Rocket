#!/usr/bin/env python3
"""Headless STL previews (matplotlib + trimesh). Writes PNGs for docs/cad-previews/."""
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


def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(str(path), force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def subsample_faces(mesh: trimesh.Trimesh, max_faces: int) -> np.ndarray:
    faces = mesh.faces
    if len(faces) <= max_faces:
        return faces
    idx = np.linspace(0, len(faces) - 1, max_faces, dtype=int)
    return faces[idx]


def render_views(mesh: trimesh.Trimesh, out_png: Path, title: str, max_faces: int = 12000) -> None:
    mesh = mesh.copy()
    mesh.apply_scale(1000.0)  # m -> mm for readable axes
    faces = subsample_faces(mesh, max_faces)
    verts = mesh.vertices
    polys = verts[faces]

    views = [
        ("side (XZ)", 0, -90, 0),
        ("top (XY)", 90, 0, 0),
        ("iso", 25, -55, 0),
    ]

    fig = plt.figure(figsize=(12, 4), dpi=150)
    fig.suptitle(title, fontsize=11)

    for i, (name, elev, azim, roll) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        coll = Poly3DCollection(polys, linewidths=0.08, alpha=0.95)
        coll.set_facecolor("#4a90d9")
        coll.set_edgecolor("#1a3a5c")
        ax.add_collection3d(coll)

        lim = np.abs(verts).max() * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title(name, fontsize=9)
        ax.view_init(elev=elev, azim=azim, roll=roll)
        ax.set_box_aspect((1, 1, 1))

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Render STL triptych previews to PNG")
    p.add_argument("--stl", required=True, help="Input STL path")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--title", default="", help="Figure title")
    p.add_argument("--max-faces", type=int, default=12000)
    args = p.parse_args()

    stl = Path(args.stl)
    if not stl.is_file():
        print(f"Missing STL: {stl}", file=sys.stderr)
        return 1

    mesh = load_mesh(stl)
    title = args.title or stl.stem.replace("_", " ")
    render_views(mesh, Path(args.out), title, max_faces=args.max_faces)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
