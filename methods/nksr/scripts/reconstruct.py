#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch
import trimesh
import nksr

# Prefer Open3D for point cloud + normals I/O (PLY point clouds often load as trimesh.PointCloud
# where normals aren't exposed as vertex_normals).
try:
    import open3d as o3d
except ImportError as e:
    o3d = None


def _normalize_normals(normals: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.clip(n, 1e-12, None)


def _load_pointcloud_with_normals(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a point cloud + normals from a .ply.

    Strategy:
      1) Try Open3D (recommended, reliable for PLY normals)
      2) Fall back to trimesh for cases where it *is* a mesh with vertex_normals

    Returns:
      vertices: (N, 3) float64
      normals:  (N, 3) float64

    Raises:
      RuntimeError if points or normals are missing.
    """
    # 1) Open3D path
    if o3d is not None:
        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) == 0:
            raise RuntimeError("Empty point cloud (Open3D).")

        vertices = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        if normals is None or normals.shape[0] == 0:
            raise RuntimeError(
                "No normals found (Open3D normals array is empty). "
                "Make sure your PLY actually contains normals."
            )

        if vertices.shape[0] != normals.shape[0]:
            raise RuntimeError(
                f"Points/normals count mismatch: points={vertices.shape[0]} normals={normals.shape[0]}"
            )

        return vertices, _normalize_normals(normals)

    # 2) Trimesh fallback
    geom = trimesh.load(path, process=False)

    if isinstance(geom, trimesh.Scene):
        if len(geom.geometry) == 0:
            raise RuntimeError("Empty scene (trimesh).")
        geom = trimesh.util.concatenate(tuple(geom.geometry.values()))

    # If this is a Trimesh (actual mesh), it may have vertex_normals
    if isinstance(geom, trimesh.Trimesh):
        vertices = np.asarray(geom.vertices)
        normals = np.asarray(geom.vertex_normals) if geom.vertex_normals is not None else None
        if normals is None or normals.shape[0] == 0:
            raise RuntimeError("No vertex_normals on loaded Trimesh.")
        if vertices.shape[0] != normals.shape[0]:
            raise RuntimeError(
                f"Vertices/normals mismatch: v={vertices.shape[0]} n={normals.shape[0]}"
            )
        return vertices, _normalize_normals(normals)

    # If it's a PointCloud, trimesh often won't expose normals like we want.
    raise RuntimeError(
        "Open3D is not available and trimesh did not load a mesh with vertex_normals. "
        "Install Open3D: pip install open3d"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NKSR Reconstruction (PLY point cloud + normals)")
    parser.add_argument("--input", type=str, required=True, help="Input PLY file (must contain normals)")
    parser.add_argument("--output", type=str, required=True, help="Output mesh file (e.g. .ply or .obj)")
    parser.add_argument("--detail_level", type=float, default=1.0, help="Detail level (higher is more detailed)")
    parser.add_argument("--voxel_size", type=float, default=None, help="Voxel size (alternative to detail level)")
    parser.add_argument("--mise_iter", type=int, default=1, help="Mise iterations for dual mesh extraction")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.input):
        raise SystemExit(f"Error: Input file not found: {args.input}")

    # Load point cloud + normals
    print(f"Loading point cloud with normals from: {args.input}")
    try:
        vertices, normals = _load_pointcloud_with_normals(args.input)
    except Exception as e:
        raise SystemExit(f"Error loading point cloud/normals: {e}")

    print(f"Point cloud stats: {vertices.shape[0]} points")
    print(f"Vertices shape: {vertices.shape}, Normals shape: {normals.shape}")

    # Torch tensors
    input_xyz = torch.from_numpy(vertices).float().to(device)
    input_normal = torch.from_numpy(normals).float().to(device)

    # Reconstruct
    print("Initializing Reconstructor...")
    reconstructor = nksr.Reconstructor(device)

    kwargs = {}
    if args.voxel_size is not None:
        kwargs["voxel_size"] = float(args.voxel_size)
        print(f"Reconstructing with voxel_size={kwargs['voxel_size']}")
    else:
        kwargs["detail_level"] = float(args.detail_level)
        print(f"Reconstructing with detail_level={kwargs['detail_level']}")

    try:
        field = reconstructor.reconstruct(input_xyz, input_normal, **kwargs)
        print("Extracting dual mesh...")
        mesh_out = field.extract_dual_mesh(mise_iter=args.mise_iter)
    except Exception as e:
        raise SystemExit(f"Reconstruction failed: {e}")

    # Convert to trimesh and export
    print(f"Saving to: {args.output}")

    v = mesh_out.v
    f = mesh_out.f

    if torch.is_tensor(v):
        v = v.detach().cpu().numpy()
    if torch.is_tensor(f):
        f = f.detach().cpu().numpy()

    v = np.asarray(v)
    f = np.asarray(f)

    if v.ndim != 2 or v.shape[1] != 3:
        raise SystemExit(f"Unexpected vertex array shape: {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise SystemExit(f"Unexpected face array shape: {f.shape}")

    out_mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    out_mesh.export(args.output)

    print("Done.")


if __name__ == "__main__":
    main()
