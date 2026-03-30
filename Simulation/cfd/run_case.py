#!/usr/bin/env python3
"""
OpenFOAM CFD Driver
===================
Sets up and runs an OpenFOAM case for a given rocket STL at a specific
velocity, angle of attack, and canard deflection.

Usage:
    python run_case.py --stl rocket.stl --velocity 40 --aoa 5 --output results/v40_a5
    python run_case.py --stl rocket.stl --velocity 40 --aoa 0 --canard-deflection 6 --output results/v40_cd6

Requires OpenFOAM module loaded (e.g. openfoam/v2412).
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent / "template"
NU_AIR = 1.5e-5  # kinematic viscosity of air at 20C


def compute_inlet_conditions(velocity: float, turbulence_intensity: float = 0.01):
    """Compute k and omega from freestream velocity and turbulence intensity."""
    k = 1.5 * (velocity * turbulence_intensity) ** 2
    k = max(k, 1e-6)
    omega = k ** 0.5 / (0.09 ** 0.25 * 0.001 * velocity) if velocity > 0.1 else 1.0
    return k, omega


def setup_case(stl_path: str, case_dir: str, velocity: float,
               aoa_deg: float = 0.0, canard_deg: float = 0.0,
               n_procs: int = 16, rocket_length: float = 0.587,
               body_radius: float = 0.0375,
               refinement_level: int = 1):
    """Copy template and parametrise for a specific run condition."""
    case = Path(case_dir)
    if case.exists():
        shutil.rmtree(case)

    shutil.copytree(TEMPLATE_DIR, case)

    stl_src = Path(stl_path).resolve()
    trisurface = case / "constant" / "triSurface"
    trisurface.mkdir(parents=True, exist_ok=True)

    try:
        import trimesh
        import numpy as np
        mesh = trimesh.load(str(stl_src), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        R = trimesh.transformations.rotation_matrix(-math.pi / 2, [0, 1, 0])
        mesh.apply_transform(R)
        mesh.export(str(trisurface / "rocket.stl"))
    except ImportError:
        shutil.copy2(stl_src, trisurface / "rocket.stl")

    d = 2 * body_radius
    S_ref = math.pi * body_radius ** 2

    domain_half = 15 * d
    x_min = -10 * rocket_length
    x_max = 20 * rocket_length

    cell_size_map = {1: d * 2, 2: d * 1, 3: d * 0.5}
    max_cells_map = {1: 8_000_000, 2: 16_000_000, 3: 32_000_000}
    cell_size = cell_size_map.get(refinement_level, d * 2)
    max_global_cells = max_cells_map.get(refinement_level, 8_000_000)

    nX = max(int((x_max - x_min) / cell_size), 40)
    nY = max(int(2 * domain_half / cell_size), 20)
    nZ = nY

    aoa_rad = math.radians(aoa_deg)
    Ux = velocity * math.cos(aoa_rad)
    Uy = velocity * math.sin(aoa_rad)
    Uz = 0.0

    drag_dir = f"{math.cos(aoa_rad):.8f} {math.sin(aoa_rad):.8f} 0"
    lift_dir = f"{-math.sin(aoa_rad):.8f} {math.cos(aoa_rad):.8f} 0"

    k, omega = compute_inlet_conditions(velocity)
    Re = velocity * rocket_length / NU_AIR if velocity > 0.1 else 1e5

    location_y = domain_half * 0.75

    replacements = {
        "__XMIN__": f"{x_min:.4f}",
        "__XMAX__": f"{x_max:.4f}",
        "__YMIN__": f"{-domain_half:.4f}",
        "__YMAX__": f"{domain_half:.4f}",
        "__ZMIN__": f"{-domain_half:.4f}",
        "__ZMAX__": f"{domain_half:.4f}",
        "__NX__": str(nX),
        "__NY__": str(nY),
        "__NZ__": str(nZ),
        "__UX__": f"{Ux:.4f}",
        "__UY__": f"{Uy:.4f}",
        "__UZ__": f"{Uz:.4f}",
        "__K__": f"{k:.6f}",
        "__OMEGA__": f"{omega:.4f}",
        "__MAGUINF__": f"{velocity:.4f}",
        "__LREF__": f"{rocket_length:.4f}",
        "__AREF__": f"{S_ref:.6f}",
        "__NPROCS__": str(n_procs),
        "__LOCATIONY__": f"{location_y:.4f}",
        "__DRAGDIR__": drag_dir,
        "__LIFTDIR__": lift_dir,
        "__MAXGLOBALCELLS__": str(max_global_cells),
    }

    for root, _dirs, files in os.walk(case):
        for fname in files:
            fpath = Path(root) / fname
            if fpath.suffix in ('.stl', '.eMesh', '.obj'):
                continue
            try:
                text = fpath.read_text()
            except UnicodeDecodeError:
                continue
            changed = False
            for key, val in replacements.items():
                if key in text:
                    text = text.replace(key, val)
                    changed = True
            if changed:
                fpath.write_text(text)

    meta = {
        "stl": str(stl_src),
        "velocity": velocity,
        "aoa_deg": aoa_deg,
        "canard_deg": canard_deg,
        "Re": Re,
        "k": k,
        "omega": omega,
        "n_procs": n_procs,
        "rocket_length": rocket_length,
        "body_radius": body_radius,
        "reference_area": S_ref,
    }
    (case / "case_meta.json").write_text(json.dumps(meta, indent=2))
    return case


def run_openfoam(case_dir: str, n_procs: int = 16, parallel: bool = True,
                   serial_mesh: bool = False):
    """Execute the full OpenFOAM pipeline: blockMesh -> surfaceFeatureExtract
    -> snappyHexMesh -> simpleFoam.

    If serial_mesh is True and n_procs > 1, mesh with single-process snappyHexMesh
    then decomposePar + parallel simpleFoam (more stable than parallel snappy).
    """
    case = Path(case_dir)

    def _run(cmd, label):
        print(f"[CFD] {label}: {' '.join(cmd)}", flush=True)
        log = case / f"log.{label}"
        with open(log, "w") as lf:
            proc = subprocess.run(cmd, cwd=case, stdout=lf, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print(f"[CFD] ERROR in {label} (exit {proc.returncode}). See {log}", flush=True)
            return False
        return True

    if not _run(["blockMesh"], "blockMesh"):
        return False
    if not _run(["surfaceFeatureExtract"], "surfaceFeatureExtract"):
        return False

    if serial_mesh and n_procs > 1:
        if not _run(["snappyHexMesh", "-overwrite"], "snappyHexMesh"):
            return False
        if not _run(["decomposePar"], "decomposePar_solve"):
            return False
        if not _run(["mpirun", "-np", str(n_procs), "simpleFoam", "-parallel"],
                     "simpleFoam"):
            return False
        if not _run(["reconstructPar", "-latestTime"], "reconstructPar"):
            return False
        return True

    if parallel and n_procs > 1:
        if not _run(["decomposePar", "-copyZero"], "decomposePar_mesh"):
            return False
        if not _run(["mpirun", "-np", str(n_procs), "snappyHexMesh", "-overwrite", "-parallel"],
                     "snappyHexMesh"):
            return False
        if not _run(["reconstructParMesh", "-constant"], "reconstructParMesh"):
            return False
        for d in case.glob("processor*"):
            shutil.rmtree(d)

        if not _run(["decomposePar"], "decomposePar_solve"):
            return False
        if not _run(["mpirun", "-np", str(n_procs), "simpleFoam", "-parallel"],
                     "simpleFoam"):
            return False
        if not _run(["reconstructPar", "-latestTime"], "reconstructPar"):
            return False
    else:
        if not _run(["snappyHexMesh", "-overwrite"], "snappyHexMesh"):
            return False
        if not _run(["simpleFoam"], "simpleFoam"):
            return False

    return True


COL_CD = 1
COL_CL = 4
COL_CM_PITCH = 7
COL_CM_ROLL = 8
COL_CM_YAW = 9
COL_CS = 10
NUM_COLS = 13


def extract_coefficients(case_dir: str) -> dict:
    """Parse forceCoeffs coefficient.dat for Cd, Cl, CmPitch, CmYaw, Cs.

    OpenFOAM column layout (13 columns):
      Time | Cd | Cd(f) | Cd(r) | Cl | Cl(f) | Cl(r) |
      CmPitch | CmRoll | CmYaw | Cs | Cs(f) | Cs(r)
    """
    case = Path(case_dir)
    coeff_dir = case / "postProcessing" / "forces"
    if not coeff_dir.exists():
        return {"error": "No forceCoeffs output found"}

    latest_time = sorted(coeff_dir.iterdir())[-1] if list(coeff_dir.iterdir()) else None
    if latest_time is None:
        return {"error": "No time directories in forceCoeffs"}

    coeff_file = latest_time / "coefficient.dat"
    if not coeff_file.exists():
        for f in latest_time.iterdir():
            if f.suffix == '.dat':
                coeff_file = f
                break

    if not coeff_file.exists():
        return {"error": f"No coefficient file in {latest_time}"}

    lines = coeff_file.read_text().strip().split('\n')
    data_lines = [l for l in lines if not l.startswith('#')]
    if not data_lines:
        return {"error": "No data in coefficient file"}

    rows = []
    for line in data_lines[-200:]:
        parts = line.split()
        if len(parts) >= NUM_COLS:
            try:
                rows.append([float(x) for x in parts[:NUM_COLS]])
            except ValueError:
                continue

    if not rows:
        return {"error": "Could not parse coefficient data"}

    cols = list(zip(*rows))

    def _avg(idx):
        return sum(cols[idx]) / len(cols[idx])

    result = {
        "Cd": _avg(COL_CD),
        "Cl": _avg(COL_CL),
        "CmPitch": _avg(COL_CM_PITCH),
        "CmYaw": _avg(COL_CM_YAW),
        "Cs": _avg(COL_CS),
        "Cm": _avg(COL_CM_PITCH),
        "n_averaged": len(rows),
    }

    meta_file = case / "case_meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        result.update(meta)

    return result


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenFOAM CFD for rocket")
    parser.add_argument("--stl", required=True, help="Path to rocket STL file")
    parser.add_argument("--velocity", type=float, default=40.0, help="Freestream velocity (m/s)")
    parser.add_argument("--aoa", type=float, default=0.0, help="Angle of attack (degrees)")
    parser.add_argument("--canard-deflection", type=float, default=0.0)
    parser.add_argument("--output", default="cfd_run", help="Output case directory")
    parser.add_argument("--nprocs", type=int, default=16)
    parser.add_argument("--setup-only", action="store_true", help="Only set up, don't run")
    parser.add_argument("--extract-only", action="store_true", help="Only extract results")
    parser.add_argument("--rocket-length", type=float, default=0.587)
    parser.add_argument("--body-radius", type=float, default=0.0375)
    parser.add_argument("--serial-mesh", action="store_true",
                        help="Mesh with serial snappyHexMesh, then parallel simpleFoam")
    parser.add_argument("--refinement-level", type=int, default=1, choices=[1, 2, 3],
                        help="Mesh refinement: 1=coarse, 2=medium, 3=fine")
    args = parser.parse_args()

    if args.extract_only:
        results = extract_coefficients(args.output)
        print(json.dumps(results, indent=2))
        sys.exit(0)

    case = setup_case(
        stl_path=args.stl,
        case_dir=args.output,
        velocity=args.velocity,
        aoa_deg=args.aoa,
        canard_deg=args.canard_deflection,
        n_procs=args.nprocs,
        rocket_length=args.rocket_length,
        body_radius=args.body_radius,
        refinement_level=args.refinement_level,
    )
    print(f"[CFD] Case set up at: {case}", flush=True)

    if args.setup_only:
        print("[CFD] Setup-only mode, exiting.", flush=True)
        sys.exit(0)

    success = run_openfoam(
        str(case), n_procs=args.nprocs, serial_mesh=args.serial_mesh)
    if success:
        results = extract_coefficients(str(case))
        print(json.dumps(results, indent=2))
        (case / "cfd_results.json").write_text(json.dumps(results, indent=2))
    else:
        print("[CFD] Simulation failed. Check logs in case directory.", flush=True)
        sys.exit(1)
