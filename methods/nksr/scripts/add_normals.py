import argparse
import numpy as np
import open3d as o3d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ply", required=True)
    ap.add_argument("--out_ply", required=True)
    ap.add_argument("--radius", type=float, default=0.2, help="Neighborhood radius (in your cloud units)")
    ap.add_argument("--max_nn", type=int, default=30)
    ap.add_argument("--orient_k", type=int, default=50, help="k for consistent orientation")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.in_ply)
    if len(pcd.points) == 0:
        raise RuntimeError("Loaded empty point cloud")

    # Estimate normals (local plane fit)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.radius, max_nn=args.max_nn)
    )

    # Make normals consistently oriented (important for surface reconstruction)
    pcd.orient_normals_consistent_tangent_plane(args.orient_k)

    # Optional: normalize
    pcd.normalize_normals()

    o3d.io.write_point_cloud(args.out_ply, pcd, write_ascii=False, compressed=False)
    print(f"Wrote: {args.out_ply} with normals")

if __name__ == "__main__":
    main()
