#!/usr/bin/env python3
"""
Preprocess a turntable / manually-rotated-object dataset for Factored-NeuS using:
- sleeve masks for COLMAP pose estimation
- connector masks for Factored-NeuS training

Input:
  <scene_dir>/
    image/                  (JPG/PNG photos)
    masques_sleeve/         (JPG/PNG sleeve masks)
    masques_connecteur/     (JPG/PNG connector masks)

Output:
  <public_data_root>/<case_name>/
    image/*.png
    mask/*.png
    cameras_sphere.npz
    preprocess_report.json
    colmap/
      database.db
      masks/*.png
      sparse/0/{cameras,images,points3D}.{bin,txt}

Important:
- This script REFUSES to use the same directory for input and output.
- It CLEANS the output case folders before writing, to avoid stale JPG/PNG mixing.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

try:
    from colmap.read_write_model import read_model, qvec2rotmat
except Exception:
    print("ERROR: Cannot import colmap.read_write_model from this repo.", file=sys.stderr)
    print("Run this script from the Factored-NeuS repo root, or add repo root to PYTHONPATH.", file=sys.stderr)
    raise


def log(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str, code: int = 2) -> None:
    print(f"FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def run(cmd: List[str], cwd: Path = None) -> None:
    log("\n>>> " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def list_images_any_ext(folder: Path) -> List[Path]:
    exts = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
    files: List[Path] = []
    for pat in exts:
        files.extend(folder.glob(pat))
    return sorted(files)


def basename_no_ext(p: Path) -> str:
    return p.stem


def build_basename_index(folder: Path) -> Dict[str, Path]:
    files = list_images_any_ext(folder)
    index: Dict[str, Path] = {}
    for f in files:
        b = basename_no_ext(f)
        if b in index:
            die(
                f"Duplicate basename '{b}' in {folder}. "
                f"Found both {index[b].name} and {f.name}."
            )
        index[b] = f
    return index


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        die(f"Failed to read grayscale image: {path}")
    return img


def read_color(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        die(f"Failed to read color image: {path}")
    return img


def write_png(path: Path, img: np.ndarray) -> None:
    ensure_parent(path)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        die(f"Failed to write PNG: {path}")


def binarize_mask(gray: np.ndarray, thr: int = 127) -> np.ndarray:
    _, m = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    return m


def colmap_check_available() -> None:
    try:
        subprocess.run(
            ["colmap", "-h"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        die("COLMAP not found on PATH. Install COLMAP or ensure `colmap` is callable.")


def colmap_command_supports_option(command: str, option: str) -> bool:
    try:
        res = subprocess.run(
            ["colmap", command, "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return option in res.stdout
    except Exception:
        return False


def camera_to_K(camera) -> np.ndarray:
    model = camera.model
    p = camera.params

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    elif model == "PINHOLE":
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif model in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    elif model in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        die(f"Unsupported camera model '{model}'. Add support in camera_to_K().")

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K


def make_world_mat(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    P = K @ np.hstack([R, t.reshape(3, 1)])
    M = np.eye(4, dtype=np.float32)
    M[:3, :4] = P.astype(np.float32)
    return M


def compute_scale_mat(
    points_xyz: np.ndarray,
    scale_margin: float = 1.10,
    radius_quantile: float = 0.99,
) -> Tuple[np.ndarray, Dict]:
    if points_xyz.shape[0] < 10:
        die("Too few sparse points to compute stable normalization.")

    xyz = points_xyz.astype(np.float64)
    pmin = xyz.min(axis=0)
    pmax = xyz.max(axis=0)
    center = 0.5 * (pmin + pmax)

    d = np.linalg.norm(xyz - center[None, :], axis=1)
    radius = float(np.quantile(d, radius_quantile))
    scale = float(radius * scale_margin)

    S = np.eye(4, dtype=np.float32)
    S[0, 0] = S[1, 1] = S[2, 2] = scale
    S[:3, 3] = center.astype(np.float32)

    meta = {
        "bbox_min": pmin.tolist(),
        "bbox_max": pmax.tolist(),
        "center": center.tolist(),
        "radius_quantile": float(radius_quantile),
        "radius": radius,
        "scale_margin": float(scale_margin),
        "scale": scale,
    }
    return S, meta


def path_eq(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except Exception:
        return a.absolute() == b.absolute()


def keep_main_mask_component(mask: np.ndarray, bin_threshold: int = 10, min_area: int = 500) -> np.ndarray:
    """
    Keep the most plausible foreground component in a binary-ish mask.
    Preference is given to large components near the image center.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D grayscale mask, got shape {mask.shape}")

    H, W = mask.shape[:2]
    cx_img = W / 2.0
    cy_img = H / 2.0

    binary = (mask > bin_threshold).astype(np.uint8) * 255

    # Clean tiny specks / fill small gaps
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return binary

    best_label = None
    best_score = None

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        cx, cy = centroids[label]
        dist2 = (cx - cx_img) ** 2 + (cy - cy_img) ** 2

        # Bigger is better; farther from center is worse
        score = area - 0.001 * dist2

        if best_score is None or score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return np.zeros_like(binary)

    cleaned = np.where(labels == best_label, 255, 0).astype(np.uint8)
    return cleaned


def compute_global_crop_box(
    common: List[str],
    img_idx: Dict[str, Path],
    sleeve_idx: Dict[str, Path],
    conn_idx: Dict[str, Path],
    crop_mask_threshold: int,
    crop_margin: float,
    crop_make_square: bool,
    mask_min_area: int,
) -> Tuple[int, int, int, int, Dict]:
    """
    Compute one global crop box from cleaned sleeve+connector masks over all images.
    Returns (x0, y0, x1, y1) in pixel coordinates, clipped to image bounds.
    """
    mins = []
    maxs = []

    first_img = read_color(img_idx[common[0]])
    H, W = first_img.shape[:2]

    for b in common:
        sleeve = keep_main_mask_component(
            read_gray(sleeve_idx[b]),
            bin_threshold=crop_mask_threshold,
            min_area=mask_min_area,
        )
        conn = keep_main_mask_component(
            read_gray(conn_idx[b]),
            bin_threshold=crop_mask_threshold,
            min_area=mask_min_area,
        )

        if sleeve.shape != conn.shape:
            die(f"Mask size mismatch for {b}: sleeve={sleeve.shape}, connector={conn.shape}")

        union = np.maximum(sleeve, conn)
        ys, xs = np.where(union > 0)

        if len(xs) == 0:
            continue

        mins.append([xs.min(), ys.min()])
        maxs.append([xs.max(), ys.max()])

    if not mins:
        die("Could not compute crop box: all cleaned union masks appear empty.")

    mins = np.array(mins)
    maxs = np.array(maxs)

    xmin = int(mins[:, 0].min())
    ymin = int(mins[:, 1].min())
    xmax = int(maxs[:, 0].max())
    ymax = int(maxs[:, 1].max())

    raw_bbox = [xmin, ymin, xmax, ymax]

    bw = xmax - xmin
    bh = ymax - ymin

    xmin = int(np.floor(xmin - bw * crop_margin))
    xmax = int(np.ceil(xmax + bw * crop_margin))
    ymin = int(np.floor(ymin - bh * crop_margin))
    ymax = int(np.ceil(ymax + bh * crop_margin))

    expanded_bbox = [xmin, ymin, xmax, ymax]

    if crop_make_square:
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        size = int(np.ceil(max(xmax - xmin, ymax - ymin)))

        xmin = int(np.floor(cx - size / 2))
        xmax = xmin + size
        ymin = int(np.floor(cy - size / 2))
        ymax = ymin + size

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(W, xmax)
    ymax = min(H, ymax)

    if xmin >= xmax or ymin >= ymax:
        die(f"Computed invalid crop box after clipping: {(xmin, ymin, xmax, ymax)}")

    final_bbox = [xmin, ymin, xmax, ymax]
    meta = {
        "image_size": [W, H],
        "raw_bbox": raw_bbox,
        "expanded_bbox": expanded_bbox,
        "final_bbox": final_bbox,
        "crop_margin": crop_margin,
        "crop_make_square": bool(crop_make_square),
        "crop_mask_threshold": crop_mask_threshold,
        "mask_min_area": mask_min_area,
    }
    return xmin, ymin, xmax, ymax, meta


def crop_array(arr: np.ndarray, crop_box: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if crop_box is None:
        return arr
    x0, y0, x1, y1 = crop_box
    return arr[y0:y1, x0:x1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True, type=str)
    ap.add_argument("--case_name", required=True, type=str)
    ap.add_argument("--public_data_root", default="./public_data", type=str)

    ap.add_argument("--colmap_matcher", default="exhaustive", choices=["exhaustive", "sequential"])
    ap.add_argument("--colmap_camera_model", default="SIMPLE_RADIAL", type=str)
    ap.add_argument("--single_camera", default=1, type=int, choices=[0, 1])
    ap.add_argument("--run_colmap", default=1, type=int, choices=[0, 1])

    ap.add_argument("--mask_threshold", default=127, type=int)
    ap.add_argument("--mask_min_area", default=500, type=int,
                    help="Minimum connected-component area kept in masks.")

    ap.add_argument("--scale_margin", default=1.10, type=float)
    ap.add_argument("--radius_quantile", default=0.99, type=float)

    # Optional crop stage
    ap.add_argument("--enable_crop", default=1, type=int, choices=[0, 1],
                    help="Crop images and masks before COLMAP/model preprocessing.")
    ap.add_argument("--crop_margin", default=0.20, type=float,
                    help="Relative margin added around the union mask bounding box.")
    ap.add_argument("--crop_mask_threshold", default=10, type=int,
                    help="Threshold used when computing crop bbox from masks.")
    ap.add_argument("--crop_make_square", default=1, type=int, choices=[0, 1],
                    help="Make the crop box square.")

    # COLMAP robustness knobs
    ap.add_argument("--colmap_guided_matching", default=1, type=int, choices=[0, 1],
                    help="Enable guided matching during COLMAP matching.")
    ap.add_argument("--colmap_estimate_affine_shape", default=1, type=int, choices=[0, 1],
                    help="Enable affine-shape SIFT extraction.")
    ap.add_argument("--colmap_domain_size_pooling", default=1, type=int, choices=[0, 1],
                    help="Enable domain-size pooling in SIFT extraction.")
    ap.add_argument("--colmap_sequential_overlap", default=25, type=int,
                    help="Sequential matcher overlap window.")
    ap.add_argument("--colmap_max_num_features", default=8192, type=int,
                    help="Optional SIFT max_num_features; useful for small textured objects.")
    ap.add_argument("--colmap_max_image_size", default=3200, type=int,
                    help="Optional SIFT max_image_size during feature extraction.")

    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    in_img_dir = scene_dir / "image"
    in_sleeve_dir = scene_dir / "masques_sleeve"
    in_conn_dir = scene_dir / "masques_connecteur"

    for d in [in_img_dir, in_sleeve_dir, in_conn_dir]:
        if not d.exists():
            die(f"Missing folder: {d}")

    out_case = (Path(args.public_data_root) / args.case_name).resolve()
    out_image = out_case / "image"
    out_mask = out_case / "mask"
    colmap_ws = out_case / "colmap"
    colmap_db = colmap_ws / "database.db"
    colmap_sparse = colmap_ws / "sparse"
    colmap_masks = colmap_ws / "masks"

    log("=== STEP 0: Validate input/output paths ===")
    log(f"scene_dir   = {scene_dir}")
    log(f"out_case    = {out_case}")

    if path_eq(scene_dir, out_case):
        die(
            "Input scene_dir and output out_case resolve to the SAME folder.\n"
            "That will mix original JPG inputs with generated PNG outputs and break COLMAP.\n"
            "Use a different output location."
        )

    if str(out_case).startswith(str(scene_dir) + "/"):
        die("Output folder is INSIDE the input scene_dir. That is unsafe.")

    log("\n=== STEP 1: Index inputs by basename ===")
    img_idx = build_basename_index(in_img_dir)
    sleeve_idx = build_basename_index(in_sleeve_dir)
    conn_idx = build_basename_index(in_conn_dir)

    log(f"Found images           : {len(img_idx)}")
    log(f"Found sleeve masks     : {len(sleeve_idx)}")
    log(f"Found connector masks  : {len(conn_idx)}")

    common = sorted(set(img_idx.keys()) & set(sleeve_idx.keys()) & set(conn_idx.keys()))
    missing_sleeve = sorted(set(img_idx.keys()) - set(sleeve_idx.keys()))
    missing_conn = sorted(set(img_idx.keys()) - set(conn_idx.keys()))

    log(f"Basenames present in all three: {len(common)}")
    if missing_sleeve:
        log(f"WARNING: missing sleeve masks for {len(missing_sleeve)} images")
        log("First 10: " + ", ".join(missing_sleeve[:10]))
    if missing_conn:
        log(f"WARNING: missing connector masks for {len(missing_conn)} images")
        log("First 10: " + ", ".join(missing_conn[:10]))

    if len(common) < 8:
        die("Too few matched triplets (image + sleeve + connector). Fix naming consistency first.")

    log("\nExample basenames (first 10):")
    for b in common[:10]:
        log(f"  {b}")

    crop_box = None
    crop_meta = None
    if args.enable_crop:
        log("\n=== STEP 1.5: Compute global crop box from cleaned sleeve+connector masks ===")
        x0, y0, x1, y1, crop_meta = compute_global_crop_box(
            common=common,
            img_idx=img_idx,
            sleeve_idx=sleeve_idx,
            conn_idx=conn_idx,
            crop_mask_threshold=args.crop_mask_threshold,
            crop_margin=args.crop_margin,
            crop_make_square=bool(args.crop_make_square),
            mask_min_area=args.mask_min_area,
        )
        crop_box = (x0, y0, x1, y1)
        log(f"Crop enabled: final crop box = {crop_box}")
        log(f"Original image size = {crop_meta['image_size']}")
        log(f"Raw bbox            = {crop_meta['raw_bbox']}")
        log(f"Expanded bbox       = {crop_meta['expanded_bbox']}")
        log(f"Final bbox          = {crop_meta['final_bbox']}")
    else:
        log("\n=== STEP 1.5: Crop disabled ===")

    log("\n=== STEP 2: Clean output case and write working dataset ===")
    ensure_clean_dir(out_case)
    ensure_clean_dir(out_image)
    ensure_clean_dir(out_mask)
    ensure_clean_dir(colmap_ws)
    ensure_clean_dir(colmap_masks)
    ensure_clean_dir(colmap_sparse)

    written = 0
    for b in common:
        img = read_color(img_idx[b])
        conn_m = read_gray(conn_idx[b])

        if conn_m.shape[:2] != img.shape[:2]:
            die(
                f"Connector mask size mismatch for {b}: "
                f"image={img.shape[:2]} mask={conn_m.shape[:2]}"
            )

        img = crop_array(img, crop_box)
        conn_m = crop_array(conn_m, crop_box)
        conn_m = keep_main_mask_component(
            conn_m,
            bin_threshold=args.mask_threshold,
            min_area=args.mask_min_area,
        )

        img_out = out_image / f"{b}.png"
        mask_out = out_mask / f"{b}.png"

        write_png(img_out, img)
        write_png(mask_out, conn_m)
        written += 1

    log(f"Wrote {written} working images to : {out_image}")
    log(f"Wrote {written} training masks to : {out_mask}")

    log("\n=== STEP 3: Prepare sleeve masks for COLMAP ===")
    sleeve_written = 0
    for b in common:
        sleeve_m = read_gray(sleeve_idx[b])
        img = read_color(out_image / f"{b}.png")

        sleeve_m = crop_array(sleeve_m, crop_box)
        sleeve_m = keep_main_mask_component(
            sleeve_m,
            bin_threshold=args.mask_threshold,
            min_area=args.mask_min_area,
        )

        if sleeve_m.shape[:2] != img.shape[:2]:
            die(
                f"Sleeve mask size mismatch for {b}: "
                f"image={img.shape[:2]} sleeve_mask={sleeve_m.shape[:2]}"
            )

        colmap_mask_name = f"{b}.png.png"
        write_png(colmap_masks / colmap_mask_name, sleeve_m)
        sleeve_written += 1

    log(f"Wrote {sleeve_written} COLMAP sleeve masks to: {colmap_masks}")

    log("\n=== STEP 4: Verify working dataset structure ===")
    out_imgs = sorted(out_image.glob("*.png"))
    out_msks = sorted(out_mask.glob("*.png"))
    colmap_msks = sorted(colmap_masks.glob("*.png"))

    log(f"Working training images (*.png) : {len(out_imgs)}")
    log(f"Working training masks  (*.png) : {len(out_msks)}")
    log(f"Working COLMAP masks    (*.png) : {len(colmap_msks)}")

    if len(out_imgs) != len(out_msks):
        die("Training image count != training mask count after conversion.")
    if len(out_imgs) != len(colmap_msks):
        die("Training image count != COLMAP sleeve mask count after conversion.")
    if len(out_imgs) == 0:
        die("No output images written.")

    log("\nFirst 10 training image / training mask pairs:")
    for i in range(min(10, len(out_imgs), len(out_msks))):
        log(f"  {out_imgs[i].name} | {out_msks[i].name}")

    log("\nFirst 10 training image -> expected COLMAP mask names:")
    for i in range(min(10, len(out_imgs))):
        log(f"  {out_imgs[i].name} -> {out_imgs[i].name}.png")

    sample_conn_mask = read_gray(out_msks[0])
    uniq_conn = np.unique(sample_conn_mask)
    log(f"\nSample connector mask unique values: {uniq_conn.tolist()}")
    if not set(uniq_conn.tolist()).issubset({0, 255}):
        die("Connector masks are not binary after conversion.")

    sample_sleeve_mask = read_gray(colmap_msks[0])
    uniq_sleeve = np.unique(sample_sleeve_mask)
    log(f"Sample sleeve mask unique values   : {uniq_sleeve.tolist()}")
    if not set(uniq_sleeve.tolist()).issubset({0, 255}):
        die("Sleeve masks are not binary after conversion.")

    if args.run_colmap == 0:
        log("\nSkipping COLMAP (--run_colmap 0). Dataset conversion complete.")
        return

    colmap_check_available()

    log("\n=== STEP 5: Run COLMAP with sleeve masks ===")
    if colmap_db.exists():
        colmap_db.unlink()
    ensure_clean_dir(colmap_sparse)

    log("COLMAP settings:")
    log(f"  matcher                    = {args.colmap_matcher}")
    log(f"  camera_model               = {args.colmap_camera_model}")
    log(f"  single_camera              = {args.single_camera}")
    log(f"  guided_matching            = {args.colmap_guided_matching}")
    log(f"  estimate_affine_shape      = {args.colmap_estimate_affine_shape}")
    log(f"  domain_size_pooling        = {args.colmap_domain_size_pooling}")
    log(f"  sequential_overlap         = {args.colmap_sequential_overlap}")
    log(f"  max_num_features           = {args.colmap_max_num_features}")
    log(f"  max_image_size             = {args.colmap_max_image_size}")

    run([
        "colmap", "feature_extractor",
        "--database_path", str(colmap_db),
        "--image_path", str(out_image),
        "--ImageReader.mask_path", str(colmap_masks),
        "--ImageReader.camera_model", args.colmap_camera_model,
        "--ImageReader.single_camera", str(args.single_camera),
        "--SiftExtraction.estimate_affine_shape", str(args.colmap_estimate_affine_shape),
        "--SiftExtraction.domain_size_pooling", str(args.colmap_domain_size_pooling),
        "--SiftExtraction.max_num_features", str(args.colmap_max_num_features),
        "--SiftExtraction.max_image_size", str(args.colmap_max_image_size),
    ])

    matcher_cmd = "exhaustive_matcher" if args.colmap_matcher == "exhaustive" else "sequential_matcher"

    if colmap_command_supports_option(matcher_cmd, "--FeatureMatching.guided_matching"):
        guided_flag = "--FeatureMatching.guided_matching"
    elif colmap_command_supports_option(matcher_cmd, "--SiftMatching.guided_matching"):
        guided_flag = "--SiftMatching.guided_matching"
    else:
        guided_flag = None
        log("WARNING: This COLMAP build does not expose a guided matching flag for the matcher command.")

    if args.colmap_matcher == "exhaustive":
        cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(colmap_db),
        ]
        if guided_flag is not None:
            cmd += [guided_flag, str(args.colmap_guided_matching)]
        run(cmd)
    else:
        cmd = [
            "colmap", "sequential_matcher",
            "--database_path", str(colmap_db),
            "--SequentialMatching.overlap", str(args.colmap_sequential_overlap),
        ]
        if guided_flag is not None:
            cmd += [guided_flag, str(args.colmap_guided_matching)]
        run(cmd)

    run([
        "colmap", "mapper",
        "--database_path", str(colmap_db),
        "--image_path", str(out_image),
        "--output_path", str(colmap_sparse),
        "--Mapper.multiple_models", "0",
    ])

    model0 = colmap_sparse / "0"
    if not model0.exists():
        die("COLMAP did not create sparse/0. Check mapper logs above.")

    log("\n=== STEP 6: Export COLMAP model to TXT for debugging ===")
    model_txt = colmap_sparse / "0_txt"
    ensure_clean_dir(model_txt)
    run([
        "colmap", "model_converter",
        "--input_path", str(model0),
        "--output_path", str(model_txt),
        "--output_type", "TXT",
    ])

    log("\n=== STEP 7: Read COLMAP model and verify filename matching ===")
    cameras, images, points3D = read_model(str(model0), ext=".bin")

    registered_names = sorted(im.name for im in images.values())
    log(f"Registered images in COLMAP model: {len(registered_names)}")
    log("First 10 registered names:")
    for n in registered_names[:10]:
        log(f"  {n}")

    train_names = [p.name for p in out_imgs]
    name_to_image = {im.name: im for im in images.values()}

    missing_in_model = [n for n in train_names if n not in name_to_image]
    extra_in_model = [n for n in registered_names if n not in set(train_names)]

    log(f"Working dataset images : {len(train_names)}")
    log(f"Registered COLMAP names: {len(registered_names)}")
    log(f"Missing in COLMAP      : {len(missing_in_model)}")
    log(f"Extra in COLMAP        : {len(extra_in_model)}")

    if missing_in_model:
        log("First 10 working images not registered by COLMAP:")
        for n in missing_in_model[:10]:
            log(f"  {n}")

    if extra_in_model:
        log("First 10 registered COLMAP names not in working dataset:")
        for n in extra_in_model[:10]:
            log(f"  {n}")

    if len(registered_names) < int(0.8 * len(train_names)):
        log("WARNING: Fewer than 80% of working images registered. Training may be unstable.")

    keep = [n for n in train_names if n in name_to_image]
    if len(keep) < 8:
        die("Too few registered images to proceed.")

    if len(keep) != len(train_names):
        log("\nFiltering working dataset to registered images only...")
        keep_set = set(Path(n).stem for n in keep)

        for p in out_imgs:
            if p.stem not in keep_set:
                p.unlink()
        for p in out_msks:
            if p.stem not in keep_set:
                p.unlink()

        out_imgs = sorted(out_image.glob("*.png"))
        out_msks = sorted(out_mask.glob("*.png"))
        train_names = [p.name for p in out_imgs]

        log("Final training names:")
        for n in train_names:
            log(f"  {n}")

        log(f"After filtering: images={len(out_imgs)} masks={len(out_msks)}")

    log("\n=== STEP 8: Build cameras_sphere.npz ===")
    pts = np.array([p.xyz for p in points3D.values()], dtype=np.float64)
    if pts.shape[0] == 0:
        die("No sparse 3D points in COLMAP model.")

    scale_mat, scale_meta = compute_scale_mat(
        pts,
        scale_margin=args.scale_margin,
        radius_quantile=args.radius_quantile,
    )

    log("Scale normalization:")
    log(f"  center           = {scale_meta['center']}")
    log(f"  radius_quantile  = {scale_meta['radius_quantile']}")
    log(f"  radius           = {scale_meta['radius']:.6f}")
    log(f"  scale_margin     = {scale_meta['scale_margin']}")
    log(f"  scale            = {scale_meta['scale']:.6f}")

    cam_dict = {}
    cams_norm = []
    cams_norm_from_P = []
    inv_scale = np.linalg.inv(scale_mat)

    for idx, name in enumerate(train_names):
        im = name_to_image[name]
        cam = cameras[im.camera_id]

        K = camera_to_K(cam).astype(np.float64)
        R = qvec2rotmat(im.qvec).astype(np.float64)
        t = im.tvec.astype(np.float64)

        world_mat = make_world_mat(K, R, t)
        cam_dict[f"world_mat_{idx}"] = world_mat.astype(np.float32)
        cam_dict[f"scale_mat_{idx}"] = scale_mat.astype(np.float32)

        Cw = (-R.T @ t.reshape(3, 1)).reshape(3)
        Cn = (inv_scale @ np.array([Cw[0], Cw[1], Cw[2], 1.0]))[:3]
        cams_norm.append(Cn)

        P = (world_mat @ scale_mat)[:3, :4].astype(np.float64)
        decomp = cv2.decomposeProjectionMatrix(P)
        C = np.asarray(decomp[2]).reshape(-1)

        if C.shape[0] == 4:
            if abs(C[3]) < 1e-12:
                die(f"Homogeneous camera center has near-zero w for {name}: {C}")
            C = C[:3] / C[3]
        elif C.shape[0] != 3:
            die(f"Unexpected camera center shape for {name}: {C.shape}, values={C}")

        cams_norm_from_P.append(C.astype(np.float64))

    cams_norm = np.stack(cams_norm, axis=0)
    cams_norm_from_P = np.stack(cams_norm_from_P, axis=0)
    delta = np.linalg.norm(cams_norm - cams_norm_from_P, axis=1)

    log(
        "NPZ pose consistency check "
        f"(normalized camera center from COLMAP vs cv2.decompose(P)): "
        f"mean={delta.mean():.3e}, max={delta.max():.3e}"
    )

    cam_file = out_case / "cameras_sphere.npz"
    np.savez(str(cam_file), **cam_dict)
    log(f"Saved: {cam_file}")

    log("\n=== STEP 9: Normalized-space sanity checks ===")
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (inv_scale @ pts_h.T).T[:, :3]
    pmin = pts_norm.min(axis=0)
    pmax = pts_norm.max(axis=0)
    cam_r = np.linalg.norm(cams_norm, axis=1)

    log(f"Sparse points bbox normalized: min={pmin.tolist()}, max={pmax.tolist()}")
    log(
        "Camera center radius normalized: "
        f"min={cam_r.min():.3f}, mean={cam_r.mean():.3f}, max={cam_r.max():.3f}"
    )

    report = {
        "case": args.case_name,
        "scene_dir": str(scene_dir),
        "out_case": str(out_case),
        "n_input_common_triplets": len(common),
        "n_registered_images": len(registered_names),
        "n_final_training_images": len(train_names),
        "crop_enabled": bool(args.enable_crop),
        "crop_meta": crop_meta,
        "colmap_camera_model": args.colmap_camera_model,
        "colmap_matcher": args.colmap_matcher,
        "single_camera": args.single_camera,
        "colmap_guided_matching": args.colmap_guided_matching,
        "colmap_estimate_affine_shape": args.colmap_estimate_affine_shape,
        "colmap_domain_size_pooling": args.colmap_domain_size_pooling,
        "colmap_sequential_overlap": args.colmap_sequential_overlap,
        "colmap_max_num_features": args.colmap_max_num_features,
        "colmap_max_image_size": args.colmap_max_image_size,
        "mask_min_area": args.mask_min_area,
        "scale_meta": scale_meta,
        "camera_center_delta_mean": float(delta.mean()),
        "camera_center_delta_max": float(delta.max()),
        "missing_in_model": missing_in_model,
        "extra_in_model": extra_in_model,
        "final_train_names": train_names,
    }

    with open(out_case / "preprocess_report.json", "w") as f:
        json.dump(report, f, indent=2)

    log(f"Saved report: {out_case / 'preprocess_report.json'}")

    log("\nDone.")
    log("Next: train Factored-NeuS or PET-NeuS using the final working dataset.")


if __name__ == "__main__":
    main()