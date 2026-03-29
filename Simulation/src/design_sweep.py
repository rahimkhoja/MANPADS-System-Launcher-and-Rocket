#!/usr/bin/env python3
"""
Aerodynamic Design Sweep
========================
Evaluates thousands of rocket configurations using the Barrowman method,
ranks them by stability + roll authority, and exports the top candidates
as parametric STL files for CFD validation.

Usage:
    python design_sweep.py --output results/sweep_results.json
    python design_sweep.py --output results/sweep_results.json --generate-stl results/stl/
"""

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from barrowman import (
    RocketGeometry, NoseCone, BodyTube, FinSet, CanardSet,
    compute_barrowman, AeroResult
)


# ── Sweep Parameter Ranges ───────────────────────────────────────────

# Constraints from the physical system
BODY_RADIUS = 0.0375       # m (fixed, 75mm OD tube)
BODY_LENGTH = 0.549         # m (fixed, existing tube)
LAUNCHER_TUBE_ID = 0.095    # m (95mm from launcher_tube.stl)
# Fins are folding: they fold flat inside the tube and deploy after launch.
# Max deployed radius is a practical/aero limit, not a tube-fit constraint.
MAX_DEPLOYED_RADIUS = 0.100  # m (200mm OD deployed, reasonable for this scale)

CG_FROM_NOSE = 0.35         # m (default; can add nose weight to move forward)
ROCKET_MASS = 0.200          # kg

PARAM_GRID = {
    "nose_shape": ["conical", "ogive", "vonkarman"],
    "nose_length": [0.060, 0.080, 0.100, 0.120],  # m (longer nose = lower drag + moves CP fwd)
    "canard_span": [0.010, 0.015, 0.020, 0.025, 0.030],  # m
    "canard_root_chord": [0.025, 0.035, 0.045],  # m
    "canard_position": [0.06, 0.08, 0.10],  # m from nose tip
    "fin_span": [0.030, 0.040, 0.050, 0.060],  # m
    "fin_root_chord": [0.050, 0.070, 0.090],  # m
    "fin_sweep": [30.0, 45.0, 60.0],  # deg
    "boat_tail": [False, True],
}


def evaluate_config(params: dict) -> dict:
    """Run Barrowman analysis on a single configuration."""
    geom = RocketGeometry(
        nose=NoseCone(
            shape=params["nose_shape"],
            length=params["nose_length"],
            base_radius=BODY_RADIUS,
        ),
        body=BodyTube(
            length=BODY_LENGTH,
            radius=BODY_RADIUS,
            boat_tail=params["boat_tail"],
            boat_tail_length=0.03,
            boat_tail_end_radius=0.030,
        ),
        fins=FinSet(
            count=4,
            root_chord=params["fin_root_chord"],
            tip_chord=params["fin_root_chord"] * 0.5,
            span=params["fin_span"],
            sweep_angle=params["fin_sweep"],
            thickness=0.002,
            position_from_nose=BODY_LENGTH + params["nose_length"] - params["fin_root_chord"] - 0.01,
        ),
        canards=CanardSet(
            count=4,
            root_chord=params["canard_root_chord"],
            tip_chord=params["canard_root_chord"] * 0.5,
            span=params["canard_span"],
            sweep_angle=0.0,
            thickness=0.001,
            position_from_nose=params["canard_position"],
        ),
    )

    result = compute_barrowman(geom, cg_from_nose=CG_FROM_NOSE)

    # Check constraints: fins fold for tube, but deployed radius must be practical
    max_outer_radius = BODY_RADIUS + max(params["fin_span"], params["canard_span"])
    fits_tube = max_outer_radius <= MAX_DEPLOYED_RADIUS

    return {
        "params": params,
        "CNa_total": result.CNa_total,
        "CP_mm": result.CP_total * 1000,
        "stability_cal": result.stability_cal,
        "Cd0": result.Cd0,
        "roll_authority": result.roll_authority,
        "fits_tube": fits_tube,
        "max_outer_radius_mm": max_outer_radius * 1000,
        "CNa_nose": result.CNa_nose,
        "CNa_fins": result.CNa_fins,
        "CNa_canards": result.CNa_canards,
    }


def score_config(r: dict) -> float:
    """Composite score: higher is better.

    Priorities:
    1. Must fit launcher tube
    2. Stability margin >= 1.5 cal
    3. Maximise roll authority
    4. Minimise drag
    """
    if not r["fits_tube"]:
        return -1e6

    stability = r["stability_cal"]
    if stability < 1.0:
        return -1e3 + stability

    # Normalised components
    stab_score = min(stability / 2.0, 1.5)           # cap at 3.0 cal
    roll_score = r["roll_authority"] * 1e4             # scale up small values
    drag_penalty = r["Cd0"] * 2.0                      # lower is better

    return stab_score + roll_score - drag_penalty


def run_sweep(param_grid: dict = None) -> list:
    """Exhaustive grid sweep over all parameter combinations."""
    if param_grid is None:
        param_grid = PARAM_GRID

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    results = []
    total = 1
    for v in values:
        total *= len(v)

    print(f"Sweeping {total} configurations...", flush=True)

    for i, combo in enumerate(itertools.product(*values)):
        params = dict(zip(keys, combo))
        r = evaluate_config(params)
        r["score"] = score_config(r)
        results.append(r)

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{total} done", flush=True)

    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"Sweep complete. {len(results)} configs evaluated.", flush=True)
    return results


def print_top_n(results: list, n: int = 10):
    """Pretty-print the top N configurations."""
    print(f"\n{'='*80}")
    print(f"Top {n} Configurations")
    print(f"{'='*80}")

    header = (f"{'Rank':>4} {'Score':>7} {'Stab':>5} {'CNa':>5} "
              f"{'Cd0':>6} {'Roll':>8} {'Nose':>9} {'NL':>4} "
              f"{'FS':>4} {'CS':>4} {'BT':>3}")
    print(header)
    print("-" * 80)

    for i, r in enumerate(results[:n]):
        p = r["params"]
        print(f"{i+1:4d} {r['score']:7.3f} {r['stability_cal']:5.2f} "
              f"{r['CNa_total']:5.2f} {r['Cd0']:6.4f} {r['roll_authority']:8.6f} "
              f"{p['nose_shape']:>9s} {p['nose_length']*1000:4.0f} "
              f"{p['fin_span']*1000:4.0f} {p['canard_span']*1000:4.0f} "
              f"{'Y' if p['boat_tail'] else 'N':>3s}")

    print(f"\nLegend: Stab=stability(cal), NL=nose_len(mm), "
          f"FS=fin_span(mm), CS=canard_span(mm), BT=boat_tail")


def generate_stl_candidates(results: list, output_dir: str, top_n: int = 5):
    """Generate STL files for the top N candidates."""
    try:
        from rocket_geometry import generate_rocket
    except ImportError:
        print("WARNING: trimesh not available, skipping STL generation", flush=True)
        return

    os.makedirs(output_dir, exist_ok=True)

    for i, r in enumerate(results[:top_n]):
        p = r["params"]
        fname = f"candidate_{i+1:02d}_{p['nose_shape']}_fs{p['fin_span']*1000:.0f}_cs{p['canard_span']*1000:.0f}.stl"
        fpath = os.path.join(output_dir, fname)

        total_length = p["nose_length"] + BODY_LENGTH
        # Match Barrowman axial placement (aft fins, canards from nose tip)
        fin_pos = max(0.02, BODY_LENGTH - p["fin_root_chord"] - 0.01)
        canard_pos = total_length - p["canard_position"]

        mesh = generate_rocket(
            nose_shape=p["nose_shape"],
            nose_length=p["nose_length"],
            body_length=BODY_LENGTH,
            body_radius=BODY_RADIUS,
            boat_tail=p["boat_tail"],
            fin_count=4,
            fin_root=p["fin_root_chord"],
            fin_tip=p["fin_root_chord"] * 0.5,
            fin_span=p["fin_span"],
            fin_sweep=p["fin_sweep"],
            fin_thickness=0.002,
            fin_position=fin_pos,
            canard_count=4,
            canard_root=p["canard_root_chord"],
            canard_tip=p["canard_root_chord"] * 0.5,
            canard_span=p["canard_span"],
            canard_sweep=0.0,
            canard_thickness=0.001,
            canard_position=canard_pos,
        )
        mesh.export(fpath)
        bb = mesh.bounds
        dims = bb[1] - bb[0]
        print(f"  [{i+1}] {fname}: {len(mesh.faces)} faces, "
              f"{dims[0]*1000:.0f}x{dims[1]*1000:.0f}x{dims[2]*1000:.0f} mm",
              flush=True)

    print(f"Generated {min(top_n, len(results))} STL candidates in {output_dir}", flush=True)


# ── Baseline comparison ──────────────────────────────────────────────

def baseline_config() -> dict:
    """Current rocket configuration from STL measurements."""
    return {
        "nose_shape": "ogive",
        "nose_length": 0.038,
        "canard_span": 0.003,   # barely protrude (2.6mm from STL)
        "canard_root_chord": 0.020,
        "canard_position": 0.08,
        "fin_span": 0.003,
        "fin_root_chord": 0.030,
        "fin_sweep": 45.0,
        "boat_tail": False,
    }


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerodynamic design sweep")
    parser.add_argument("--output", default="results/sweep_results.json")
    parser.add_argument("--generate-stl", default=None,
                        help="Directory to output top candidate STLs")
    parser.add_argument("--from-json", default=None,
                        help="Load prior sweep JSON; skip re-sweep (fast STL export)")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top results to display/export")
    args = parser.parse_args()

    if args.from_json:
        with open(args.from_json, "r") as f:
            save_data = json.load(f)
        results = save_data.get("top_results", [])
        if not results:
            print("No top_results in JSON", flush=True)
            sys.exit(1)
        print(f"Loaded {len(results)} results from {args.from_json}", flush=True)
        print_top_n(results, args.top)
        if args.generate_stl:
            print(f"\n=== Generating Top {args.top} Candidate STLs ===", flush=True)
            generate_stl_candidates(results, args.generate_stl, args.top)
        sys.exit(0)

    # Evaluate baseline first
    print("=== Baseline Configuration ===", flush=True)
    base = evaluate_config(baseline_config())
    base["score"] = score_config(base)
    print(f"  Stability: {base['stability_cal']:.2f} cal")
    print(f"  CNa total: {base['CNa_total']:.3f}")
    print(f"  Cd0:       {base['Cd0']:.4f}")
    print(f"  Roll auth: {base['roll_authority']:.6f}")
    print(f"  Score:     {base['score']:.3f}")

    # Run sweep
    print("\n=== Design Sweep ===", flush=True)
    results = run_sweep()

    # Display top results
    print_top_n(results, args.top)

    # Compare best to baseline
    best = results[0]
    print(f"\n=== Improvement Over Baseline ===")
    print(f"  Stability: {base['stability_cal']:.2f} -> {best['stability_cal']:.2f} cal")
    print(f"  CNa:       {base['CNa_total']:.3f} -> {best['CNa_total']:.3f}")
    print(f"  Cd0:       {base['Cd0']:.4f} -> {best['Cd0']:.4f}")
    print(f"  Roll auth: {base['roll_authority']:.6f} -> {best['roll_authority']:.6f} "
          f"({best['roll_authority']/max(base['roll_authority'],1e-9):.1f}x)")

    # Save results
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    save_data = {
        "baseline": base,
        "top_results": results[:args.top * 5],
        "total_evaluated": len(results),
        "best_params": best["params"],
    }
    with open(args.output, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {args.output}", flush=True)

    # Generate STLs if requested
    if args.generate_stl:
        print(f"\n=== Generating Top {args.top} Candidate STLs ===", flush=True)
        generate_stl_candidates(results, args.generate_stl, args.top)
