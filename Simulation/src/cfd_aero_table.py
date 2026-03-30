"""
CFD Aerodynamic Lookup Table
=============================
Builds Cd(V, AoA) and Cl(V, AoA) interpolation tables from CFD sweep results.
Exports a JSON table consumable by the 6-DOF simulator.

Usage:
    python cfd_aero_table.py --input ../results/cfd_all_sweeps.json --output ../results/aero_table.json
    python cfd_aero_table.py --input-dir ../results/cfd_baseline_2079 --output ../results/aero_table.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.interpolate import RegularGridInterpolator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def parse_case_name(name: str) -> Tuple[float, float]:
    """Extract velocity and AoA from case directory name like 'v40_a5'."""
    m = re.match(r'v(\d+)_a(\d+)', name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0


def load_sweep_results(path: str) -> List[dict]:
    """Load CFD results from either all_results.json or cfd_all_sweeps.json."""
    data = json.loads(Path(path).read_text())

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        all_results = []
        for job_id, cases in data.items():
            if isinstance(cases, list):
                for r in cases:
                    r.setdefault('job_id', job_id)
                    all_results.append(r)
        return all_results

    return []


def load_from_directory(dirpath: str) -> List[dict]:
    """Load from a single CFD output directory containing all_results.json."""
    p = Path(dirpath) / "all_results.json"
    if p.exists():
        return json.loads(p.read_text())
    return []


def build_table(results: List[dict]) -> dict:
    """Build structured aero table from CFD results.

    Returns a dict with:
      velocities: sorted list of V values
      aoa_degrees: sorted list of AoA values
      Cd: 2D array [len(V) x len(AoA)]
      Cl: 2D array
      CmPitch: 2D array
    """
    points = {}
    for r in results:
        if 'error' in r:
            continue
        case = r.get('case', '')
        v = r.get('velocity', 0.0)
        aoa = r.get('aoa_deg', 0.0)
        if v == 0 and case:
            v, aoa = parse_case_name(case)
        if v <= 0:
            continue
        points[(v, aoa)] = {
            'Cd': r.get('Cd', 0.0),
            'Cl': r.get('Cl', 0.0),
            'CmPitch': r.get('CmPitch', r.get('Cm', 0.0)),
            'Cs': r.get('Cs', 0.0),
        }

    if not points:
        raise ValueError("No valid CFD data points found")

    velocities = sorted(set(k[0] for k in points))
    aoas = sorted(set(k[1] for k in points))

    table = {
        'velocities': velocities,
        'aoa_degrees': aoas,
        'Cd': [],
        'Cl': [],
        'CmPitch': [],
    }

    for v in velocities:
        cd_row, cl_row, cm_row = [], [], []
        for a in aoas:
            p = points.get((v, a))
            if p:
                cd_row.append(p['Cd'])
                cl_row.append(p['Cl'])
                cm_row.append(p['CmPitch'])
            else:
                cd_row.append(float('nan'))
                cl_row.append(float('nan'))
                cm_row.append(float('nan'))
        table['Cd'].append(cd_row)
        table['Cl'].append(cl_row)
        table['CmPitch'].append(cm_row)

    return table


class AeroTable:
    """Interpolation wrapper for CFD aero tables.

    Provides Cd(V, alpha_deg) and Cl(V, alpha_deg) lookups with bilinear
    interpolation. Falls back to nearest-neighbor outside the table bounds.
    """

    def __init__(self, table: dict):
        self.velocities = np.array(table['velocities'])
        self.aoas = np.array(table['aoa_degrees'])
        self.Cd_data = np.array(table['Cd'])
        self.Cl_data = np.array(table['Cl'])
        self.CmPitch_data = np.array(table['CmPitch'])

        if SCIPY_AVAILABLE and len(self.velocities) > 1 and len(self.aoas) > 1:
            self._Cd_interp = RegularGridInterpolator(
                (self.velocities, self.aoas), self.Cd_data,
                method='linear', bounds_error=False, fill_value=None)
            self._Cl_interp = RegularGridInterpolator(
                (self.velocities, self.aoas), self.Cl_data,
                method='linear', bounds_error=False, fill_value=None)
            self._Cm_interp = RegularGridInterpolator(
                (self.velocities, self.aoas), self.CmPitch_data,
                method='linear', bounds_error=False, fill_value=None)
        else:
            self._Cd_interp = None
            self._Cl_interp = None
            self._Cm_interp = None

    @classmethod
    def from_json(cls, path: str) -> 'AeroTable':
        data = json.loads(Path(path).read_text())
        return cls(data)

    def _lookup_nearest(self, data, V, alpha_deg):
        vi = np.argmin(np.abs(self.velocities - V))
        ai = np.argmin(np.abs(self.aoas - abs(alpha_deg)))
        return float(data[vi, ai])

    def Cd(self, V: float, alpha_deg: float) -> float:
        alpha_deg = abs(alpha_deg)
        if self._Cd_interp is not None:
            val = float(self._Cd_interp([[V, alpha_deg]])[0])
            if not np.isnan(val):
                return val
        return self._lookup_nearest(self.Cd_data, V, alpha_deg)

    def Cl(self, V: float, alpha_deg: float) -> float:
        sign = 1.0 if alpha_deg >= 0 else -1.0
        a = abs(alpha_deg)
        if self._Cl_interp is not None:
            val = float(self._Cl_interp([[V, a]])[0])
            if not np.isnan(val):
                return val * sign
        return self._lookup_nearest(self.Cl_data, V, a) * sign

    def CmPitch(self, V: float, alpha_deg: float) -> float:
        sign = 1.0 if alpha_deg >= 0 else -1.0
        a = abs(alpha_deg)
        if self._Cm_interp is not None:
            val = float(self._Cm_interp([[V, a]])[0])
            if not np.isnan(val):
                return val * sign
        return self._lookup_nearest(self.CmPitch_data, V, a) * sign


def main():
    parser = argparse.ArgumentParser(description='Build CFD aero lookup table')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to cfd_all_sweeps.json or all_results.json')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Path to CFD results directory')
    parser.add_argument('--output', type=str, default='../results/aero_table.json')
    args = parser.parse_args()

    if args.input:
        results = load_sweep_results(args.input)
    elif args.input_dir:
        results = load_from_directory(args.input_dir)
    else:
        default = Path(__file__).parent / '..' / 'results' / 'cfd_all_sweeps.json'
        if default.exists():
            results = load_sweep_results(str(default))
        else:
            print("ERROR: No input specified and cfd_all_sweeps.json not found")
            return

    print(f"Loaded {len(results)} CFD data points")

    table = build_table(results)
    print(f"Table grid: {len(table['velocities'])} velocities x {len(table['aoa_degrees'])} AoAs")
    print(f"  V  = {table['velocities']}")
    print(f"  AoA = {table['aoa_degrees']}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(table, f, indent=2)
    print(f"Saved to {args.output}")

    if SCIPY_AVAILABLE:
        at = AeroTable(table)
        print("\nSpot checks:")
        for v in table['velocities']:
            for a in table['aoa_degrees']:
                print(f"  V={v:5.0f} AoA={a:5.1f}  Cd={at.Cd(v,a):.4f}  Cl={at.Cl(v,a):.6f}")


if __name__ == "__main__":
    main()
