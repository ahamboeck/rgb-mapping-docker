import torch
import nksr
import trimesh
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="NKSR Reconstruction")
    parser.add_argument("--input", type=str, required=True, help="Input PLY file with normals")
    parser.add_argument("--output", type=str, required=True, help="Output mesh file (e.g. output.ply or output.obj)")
    parser.add_argument("--detail_level", type=float, default=1.0, help="Detail level (higher is more detailed)")
    parser.add_argument("--voxel_size", type=float, default=None, help="Voxel size (alternative to detail level)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    print(f"Loading {args.input}...")
    try:
        pcd = trimesh.load(args.input)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Handle Scene objects (common if loading .ply with trimesh)
    if isinstance(pcd, trimesh.Scene):
        if len(pcd.geometry) == 0:
            print("Error: Empty scene.")
            return
        # If multiple geometries, combine them or take the first
        print(f"Input is a Scene with {len(pcd.geometry)} geometries. Merging...")
        pcd = trimesh.util.concatenate(tuple(pcd.geometry.values()))

    # Check for normals
    if not hasattr(pcd, 'vertex_normals') or pcd.vertex_normals is None or pcd.vertex_normals.shape[0] == 0:
        print("Error: Input point cloud has no vertex normals. NKSR requires normals.")
        # Optional: Estimate normals if missing? 
        # print("Estimating normals...")
        # pcd.estimate_normals() # trimesh doesn't have a robust normal estimator built-in for simple point clouds usually, better to fail.
        return

    vertices = np.asarray(pcd.vertices)
    normals = np.asarray(pcd.vertex_normals)

    print(f"Point cloud stats: {vertices.shape[0]} points")

    input_xyz = torch.from_numpy(vertices).float().to(device)
    input_normal = torch.from_numpy(normals).float().to(device)
    
    print("Initializing Reconstructor...")
    reconstructor = nksr.Reconstructor(device)
    
    print("Reconstructing...")
    # Note: reconstruct signature might vary, but based on snippet:
    # field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=1.0)
    
    kwargs = {}
    if args.voxel_size is not None:
        kwargs['voxel_size'] = args.voxel_size
    else:
        kwargs['detail_level'] = args.detail_level

    try:
        field = reconstructor.reconstruct(input_xyz, input_normal, **kwargs)
        print("Extracting dual mesh...")
        mesh_out = field.extract_dual_mesh(mise_iter=1)
    except Exception as e:
        print(f"Reconstruction failed: {e}")
        return

    # Save output
    print(f"Saving to {args.output}...")
    
    # mesh_out is likely a structure with v and f. 
    # Let's inspect what we get or assume standard trimesh compatible format or convert.
    # If it's the internal NKSR mesh format, it usually has .v and .f (torch tensors or numpy)
    
    v = mesh_out.v
    f = mesh_out.f
    
    if torch.is_tensor(v):
        v = v.cpu().numpy()
    if torch.is_tensor(f):
        f = f.cpu().numpy()
        
    out_mesh = trimesh.Trimesh(vertices=v, faces=f)
    out_mesh.export(args.output)
    print("Done.")

if __name__ == "__main__":
    main()
