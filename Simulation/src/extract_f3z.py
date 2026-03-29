"""
F3Z Mesh Extractor
==================
Extracts mesh bodies from Fusion 360 .f3z archives.

F3Z -> ZIP of F3D files -> each F3D is a ZIP (often zstd-compressed)
containing SMB/SMBH BRep bodies in Autodesk's binary mesh format.

The SMB files contain tessellated triangle meshes. This script extracts
them and converts to STL for use with OpenFOAM / visualization.
"""

import io
import json
import os
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def read_inner_file(outer_zip, f3d_name, inner_name):
    """Read a file from inside an F3D (which is itself a ZIP inside the F3Z)."""
    with outer_zip.open(f3d_name) as f3d_raw:
        f3d_bytes = io.BytesIO(f3d_raw.read())

    with zipfile.ZipFile(f3d_bytes) as inner:
        info = inner.getinfo(inner_name)

        if info.compress_type == 93 and HAS_ZSTD:
            raw = inner.read(inner_name)  # will fail on 3.11
        elif info.compress_type == 93:
            with inner.open(inner_name) as fp:
                compressed = fp.read()
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(compressed)
        else:
            return inner.read(inner_name)


def read_inner_file_zstd(outer_zip, f3d_name, inner_name):
    """Manually handle zstd-compressed entries by reading raw bytes."""
    with outer_zip.open(f3d_name) as f3d_raw:
        f3d_bytes = f3d_raw.read()

    inner_zip = zipfile.ZipFile(io.BytesIO(f3d_bytes))
    info = inner_zip.getinfo(inner_name)

    if info.compress_type == 0:
        return inner_zip.read(inner_name)

    if info.compress_type == 93 and HAS_ZSTD:
        # Read the raw compressed bytes from the ZIP
        zf = io.BytesIO(f3d_bytes)
        zf.seek(info.header_offset)
        # Read local file header
        sig = zf.read(4)
        if sig != b'PK\x03\x04':
            raise ValueError("Bad local header")
        zf.read(22)  # skip fixed header fields
        fname_len = struct.unpack('<H', zf.read(2))[0]
        extra_len = struct.unpack('<H', zf.read(2))[0]
        zf.read(fname_len + extra_len)
        compressed = zf.read(info.compress_size)

        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(compressed, max_output_size=info.file_size)

    raise NotImplementedError(f"Compression method {info.compress_type}")


def try_parse_smb_as_mesh(data):
    """Try to interpret SMB data as a triangle mesh."""
    if not HAS_TRIMESH:
        return None

    # SMB is Autodesk's proprietary binary BRep. The tessellation
    # is sometimes embedded as triangulated faces. Try a few strategies.

    # Strategy 1: look for STL-like binary data embedded in the SMB
    stl_magic = b'solid'
    if data[:5] == stl_magic:
        try:
            mesh = trimesh.load(io.BytesIO(data), file_type='stl', force='mesh')
            if hasattr(mesh, 'vertices') and len(mesh.vertices) > 2:
                return mesh
        except Exception:
            pass

    # Strategy 2: try binary STL (80-byte header + triangle count)
    if len(data) > 84:
        try:
            mesh = trimesh.load(io.BytesIO(data), file_type='stl', force='mesh')
            if hasattr(mesh, 'vertices') and len(mesh.vertices) > 2:
                return mesh
        except Exception:
            pass

    # Strategy 3: scan for recognizable float triplet patterns
    # that look like vertex data (heuristic)
    mesh = scan_for_triangle_soup(data)
    if mesh is not None:
        return mesh

    return None


def scan_for_triangle_soup(data):
    """
    Heuristic: scan binary data for blocks of IEEE-754 floats
    that look like vertex coordinates (values in a reasonable range).
    """
    if not HAS_TRIMESH or len(data) < 100:
        return None

    # Look for patterns: many consecutive floats in [-1000, 1000] range
    n_floats = len(data) // 4
    if n_floats < 9:
        return None

    try:
        floats = np.frombuffer(data[:n_floats * 4], dtype='<f4')
    except Exception:
        return None

    # Find runs of "reasonable" floats
    reasonable = np.abs(floats) < 2000
    reasonable &= np.isfinite(floats)

    # Look for the longest run
    runs = np.diff(np.where(np.concatenate(([False], reasonable, [False])))[0])
    if len(runs) == 0:
        return None

    best_run_len = np.max(runs)
    if best_run_len < 36:  # need at least 4 triangles (36 floats = 12 vertices)
        return None

    best_run_start = np.where(runs == best_run_len)[0][0]
    starts = np.where(np.concatenate(([False], reasonable, [False])))[0]
    offset = starts[best_run_start]

    n_usable = (best_run_len // 3) * 3
    coords = floats[offset:offset + n_usable].reshape(-1, 3)

    if len(coords) < 3:
        return None

    # Treat as triangle soup (every 3 vertices = 1 triangle)
    n_tris = len(coords) // 3
    if n_tris < 2:
        return None

    verts = coords[:n_tris * 3]
    faces = np.arange(n_tris * 3).reshape(-1, 3)

    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.merge_vertices()
        if len(mesh.vertices) > 3 and len(mesh.faces) > 1:
            return mesh
    except Exception:
        pass

    return None


def extract_all(f3z_path, output_dir):
    """Extract all mesh data from an F3Z archive."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    with zipfile.ZipFile(f3z_path) as outer:
        # Find all F3D files
        f3d_files = [n for n in outer.namelist() if n.lower().endswith('.f3d')]

        for f3d_name in f3d_files:
            print(f"\nProcessing: {f3d_name}")
            with outer.open(f3d_name) as f:
                f3d_data = io.BytesIO(f.read())

            try:
                inner = zipfile.ZipFile(f3d_data)
            except zipfile.BadZipFile:
                print(f"  Not a valid ZIP, skipping")
                continue

            smb_files = [n for n in inner.namelist()
                         if n.lower().endswith('.smb') or n.lower().endswith('.smbh')]

            # Also look for images (thumbnails)
            img_files = [n for n in inner.namelist()
                         if any(n.lower().endswith(e) for e in ('.png', '.jpg', '.jpeg'))]

            for img in img_files:
                try:
                    info = inner.getinfo(img)
                    if info.compress_type == 0:
                        img_data = inner.read(img)
                    elif info.compress_type == 93 and HAS_ZSTD:
                        img_data = read_inner_file_zstd(outer, f3d_name, img)
                    else:
                        continue
                    safe = Path(f3d_name).stem + "_" + Path(img).name
                    out_path = os.path.join(output_dir, safe)
                    with open(out_path, 'wb') as fp:
                        fp.write(img_data)
                    print(f"  Image: {safe} ({len(img_data)} bytes)")
                except Exception as e:
                    print(f"  Image {img}: {e}")

            for smb_name in smb_files:
                info = inner.getinfo(smb_name)
                print(f"  SMB: {Path(smb_name).name}  "
                      f"compressed={info.compress_size}  "
                      f"size={info.file_size}  "
                      f"method={info.compress_type}")

                try:
                    if info.compress_type == 0:
                        data = inner.read(smb_name)
                    elif info.compress_type == 93 and HAS_ZSTD:
                        data = read_inner_file_zstd(outer, f3d_name, smb_name)
                    else:
                        print(f"    Unsupported compression: {info.compress_type}")
                        continue
                except Exception as e:
                    print(f"    Read error: {e}")
                    continue

                print(f"    Decompressed: {len(data)} bytes, "
                      f"header: {data[:16].hex()}")

                mesh = try_parse_smb_as_mesh(data)
                if mesh is not None:
                    safe = Path(f3d_name).stem + "_" + Path(smb_name).stem + ".stl"
                    out_path = os.path.join(output_dir, safe)
                    mesh.export(out_path)
                    print(f"    -> {safe}  "
                          f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")
                    results.append({
                        'path': out_path,
                        'source': f"{f3d_name}/{smb_name}",
                        'vertices': len(mesh.vertices),
                        'faces': len(mesh.faces),
                    })
                else:
                    # Save raw for inspection
                    raw_name = Path(f3d_name).stem + "_" + Path(smb_name).name
                    raw_path = os.path.join(output_dir, raw_name)
                    with open(raw_path, 'wb') as fp:
                        fp.write(data)
                    print(f"    No mesh found, saved raw: {raw_name}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract meshes from F3Z')
    parser.add_argument('input', help='Input .f3z file')
    parser.add_argument('-o', '--output-dir', default='./f3z_export',
                        help='Output directory')
    args = parser.parse_args()

    print(f"F3Z Mesh Extractor")
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"zstd: {HAS_ZSTD}  trimesh: {HAS_TRIMESH}")

    results = extract_all(args.input, args.output_dir)

    print(f"\n{'='*50}")
    print(f"Extracted {len(results)} mesh(es)")
    for r in results:
        print(f"  {Path(r['path']).name}: {r['vertices']} verts, {r['faces']} faces")


if __name__ == '__main__':
    main()
