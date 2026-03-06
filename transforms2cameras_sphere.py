import argparse

import json
import numpy as np
import os

# chemins
def parse_args():
    parser = argparse.ArgumentParser(description="Convert a transforms.json file to cameras_sphere.npz.")
    parser.add_argument("--transforms_path", default="transforms.json", help="transforms.json (input file) path")
    parser.add_argument("--cameras_sphere_path", default="cameras_sphere.npz", help="cameras_sphere.npz (output file) path")
    args = parser.parse_args()
    return args

args = parse_args()
transforms_path = args.transforms_path
output_path = args.cameras_sphere_path

# charge transforms.json
with open(transforms_path, "r") as f:
    meta = json.load(f)

frames = meta["frames"]

if len(frames) == 0:
    raise ValueError("transforms.json ne contient aucune frame !")

# calcul du centre et normalisation sphérique
poses = []
for frame in frames:
    pose = np.array(frame["transform_matrix"])
    poses.append(pose)
poses = np.stack(poses)

centers = poses[:, :3, 3]
center_mean = centers.mean(axis=0)
poses[:, :3, 3] -= center_mean
max_radius = np.max(np.linalg.norm(poses[:, :3, 3], axis=1))
poses[:, :3, 3] /= max_radius

# intrinsics
camera_angle_x = meta["camera_angle_x"]
W, H = frames[0].get("w", 800), frames[0].get("h", 800)  # fallback si pas dans JSON
focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

# sauvegarde par image
npz_dict = {}
for i, pose in enumerate(poses):
    npz_dict[f"world_mat_{i}"] = pose
    npz_dict[f"scale_mat_{i}"] = np.eye(4)  # scale_mat pour cette image

    K = np.eye(4)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = W / 2
    K[1, 2] = H / 2
    npz_dict[f"intrinsic_mat_{i}"] = K

np.savez(output_path, **npz_dict)
print(f"Saved {output_path} avec {len(frames)} poses.")
