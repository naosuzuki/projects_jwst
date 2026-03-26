"""
Find Supernova by Comparing JWST and HST PNG Images

Detects supernova candidates by finding bright sources in the JWST image
that have no counterpart in the HST image. Avoids image subtraction by
detecting sources independently in each image and cross-matching.

Filename conventions:
    HST:  {ID}_{Field}_hstcosmos.png   (e.g., 783487_B10_hstcosmos.png)
    JWST: {ID}_{Field}_jwst1.png       (e.g., 783487_B4_jwst1.png)
          {ID}_{Field}_jwst2.png       (e.g., 783487_B4_jwst2.png)
    Images are matched by object ID (field names may differ between HST/JWST).
    Each HST image is compared against both jwst1 and jwst2 for that object.

HST directory: /data/astrofs2_1/suzuki/data/HST/cosmosacs/original_png/
JWST directory: /data/astrofs2_1/suzuki/data/JWST/cosmosweb/v0.8_png

Usage:
    python find_supernova.py [options]
    python find_supernova.py --num-workers 16 --output-dir results/
"""

import argparse
import csv
import glob
import os
import re
import time
from multiprocessing import Pool, cpu_count

# 23 confirmed supernovae identified by visual inspection
KNOWN_SUPERNOVAE = [
    "471959", "53669", "296673", "296868", "318858", "320233",
    "9748", "53669", "52864", "407613", "239246", "239875",
    "245766", "468896", "469221", "120035", "63919", "78323",
    "25141", "25283", "27350", "435613", "435768",
]
KNOWN_SUPERNOVAE_SET = set(KNOWN_SUPERNOVAE)

import cv2
import numpy as np
from scipy.spatial import cKDTree


def load_image(filepath):
    """Load a PNG image and convert to grayscale float64."""
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {filepath}")
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return gray.astype(np.float64)


def detect_sources(image):
    """Detect bright point sources using blob detection.

    Returns array of (x, y, brightness) for each detected source.
    """
    img_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 30
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 3
    params.maxArea = 500
    params.filterByCircularity = True
    params.minCircularity = 0.3
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = True
    params.minInertiaRatio = 0.3
    params.filterByColor = True
    params.blobColor = 255

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img_8bit)

    if len(keypoints) == 0:
        return np.empty((0, 3))

    sources = []
    for kp in keypoints:
        x, y = kp.pt
        ix, iy = int(round(x)), int(round(y))
        y_lo = max(0, iy - 2)
        y_hi = min(image.shape[0], iy + 3)
        x_lo = max(0, ix - 2)
        x_hi = min(image.shape[1], ix + 3)
        brightness = np.mean(image[y_lo:y_hi, x_lo:x_hi])
        sources.append([x, y, brightness])

    return np.array(sources)


def find_new_sources(jwst_sources, hst_sources, match_radius=10.0):
    """Find JWST sources with no HST counterpart within match_radius pixels."""
    if len(jwst_sources) == 0:
        return np.empty((0, 3))
    if len(hst_sources) == 0:
        return jwst_sources

    hst_tree = cKDTree(hst_sources[:, :2])
    distances, _ = hst_tree.query(jwst_sources[:, :2])

    unmatched = distances > match_radius
    return jwst_sources[unmatched]


def save_candidates_image(image, candidates, output_path):
    """Save image with supernova candidates circled in red."""
    img_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_color = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)

    for x, y, brightness in candidates:
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(img_color, (cx, cy), 15, (0, 0, 255), 2)
        cv2.putText(img_color, f"{brightness:.0f}", (cx + 18, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imwrite(output_path, img_color)


def process_pair(hst_path, jwst_path, match_radius, min_brightness, output_dir):
    """Process a single HST/JWST image pair. Returns list of candidate tuples."""
    try:
        hst_data = load_image(hst_path)
        jwst_data = load_image(jwst_path)

        hst_sources = detect_sources(hst_data)
        jwst_sources = detect_sources(jwst_data)

        candidates = find_new_sources(jwst_sources, hst_sources, match_radius)

        if len(candidates) > 0 and min_brightness > 0:
            candidates = candidates[candidates[:, 2] >= min_brightness]

        if len(candidates) > 0:
            candidates = candidates[candidates[:, 2].argsort()[::-1]]

        if len(candidates) > 0 and output_dir:
            basename = os.path.splitext(os.path.basename(jwst_path))[0]
            out_path = os.path.join(output_dir, f"{basename}_candidates.png")
            save_candidates_image(jwst_data, candidates, out_path)

        return candidates
    except Exception as e:
        print(f"  Error processing {os.path.basename(jwst_path)}: {e}")
        return np.empty((0, 3))


def find_common_sources(sources1, sources2, match_radius=10.0):
    """Find sources that appear in both sets within match_radius pixels.

    Returns the matched sources from sources1 with their positions.
    """
    if len(sources1) == 0 or len(sources2) == 0:
        return np.empty((0, 3))

    tree2 = cKDTree(sources2[:, :2])
    distances, _ = tree2.query(sources1[:, :2])

    matched = distances <= match_radius
    return sources1[matched]


def process_object(args):
    """Worker function for multiprocessing. Processes one object.

    Supernova detection logic:
    - JWST supernova: a new source must appear in BOTH jwst1 and jwst2
      (same SN visible in both filters) but NOT in HST.
    - HST supernova: a source in HST that is NOT in jwst1 and NOT in jwst2
      (SN was active during HST epoch, faded by JWST epoch).

    Args:
        tuple: (obj_key, files_dict, match_radius, min_brightness, output_dir)

    Returns:
        list of [obj_key, hst_field, jwst_field, detection_type, x, y, brightness] rows
    """
    obj_key, files, match_radius, min_brightness, output_dir = args
    results = []

    hst_path = files.get("hst")
    jwst1_path = files.get("jwst1")
    jwst2_path = files.get("jwst2")

    try:
        # Load and detect sources in all available images
        hst_sources = np.empty((0, 3))
        jwst1_sources = np.empty((0, 3))
        jwst2_sources = np.empty((0, 3))

        if hst_path:
            hst_data = load_image(hst_path)
            hst_sources = detect_sources(hst_data)
        if jwst1_path:
            jwst1_data = load_image(jwst1_path)
            jwst1_sources = detect_sources(jwst1_data)
        if jwst2_path:
            jwst2_data = load_image(jwst2_path)
            jwst2_sources = detect_sources(jwst2_data)

        hst_field = files.get("hst_field", "")
        jwst_field = files.get("jwst_field", "")

        # Galaxy is at center of each PNG cutout.
        # Real SN only occurs near the host galaxy, so filter candidates
        # to within 30% of image center.
        def filter_near_center(sources, image_shape, max_frac=0.3):
            if len(sources) == 0:
                return sources
            h, w = image_shape
            cx, cy = w / 2.0, h / 2.0
            max_dist = max_frac * min(h, w)
            dist = np.sqrt((sources[:, 0] - cx)**2 + (sources[:, 1] - cy)**2)
            return sources[dist <= max_dist]

        # --- JWST supernova: source in BOTH jwst1 and jwst2, but NOT in HST ---
        if len(jwst1_sources) > 0 and len(jwst2_sources) > 0:
            # Find sources in jwst1 that also appear in jwst2
            jwst_common = find_common_sources(jwst1_sources, jwst2_sources, match_radius)

            # Of those, find ones NOT in HST
            if len(jwst_common) > 0 and len(hst_sources) > 0:
                candidates = find_new_sources(jwst_common, hst_sources, match_radius)
            elif len(jwst_common) > 0:
                candidates = jwst_common
            else:
                candidates = np.empty((0, 3))

            # SN must be near host galaxy (image center)
            if len(candidates) > 0:
                candidates = filter_near_center(candidates, jwst1_data.shape)
            if len(candidates) > 0 and min_brightness > 0:
                candidates = candidates[candidates[:, 2] >= min_brightness]
            if len(candidates) > 0:
                candidates = candidates[candidates[:, 2].argsort()[::-1]]

            for x, y, bright in candidates:
                results.append([obj_key, hst_field, jwst_field, "JWST_SN",
                                x, y, bright])

            if len(candidates) > 0 and output_dir and jwst1_path:
                basename = os.path.splitext(os.path.basename(jwst1_path))[0]
                out_path = os.path.join(output_dir, f"{basename}_jwst_sn.png")
                save_candidates_image(jwst1_data, candidates, out_path)

        # --- HST supernova: source in HST, but NOT in jwst1 AND NOT in jwst2 ---
        if len(hst_sources) > 0:
            # Find HST sources not in jwst1
            if len(jwst1_sources) > 0:
                hst_not_jwst1 = find_new_sources(hst_sources, jwst1_sources, match_radius)
            else:
                hst_not_jwst1 = hst_sources

            # Of those, also not in jwst2
            if len(hst_not_jwst1) > 0 and len(jwst2_sources) > 0:
                hst_candidates = find_new_sources(hst_not_jwst1, jwst2_sources, match_radius)
            else:
                hst_candidates = hst_not_jwst1

            # SN must be near host galaxy (image center)
            if len(hst_candidates) > 0:
                hst_candidates = filter_near_center(hst_candidates, hst_data.shape)
            if len(hst_candidates) > 0 and min_brightness > 0:
                hst_candidates = hst_candidates[hst_candidates[:, 2] >= min_brightness]
            if len(hst_candidates) > 0:
                hst_candidates = hst_candidates[hst_candidates[:, 2].argsort()[::-1]]

            for x, y, bright in hst_candidates:
                results.append([obj_key, hst_field, jwst_field, "HST_SN",
                                x, y, bright])

            if len(hst_candidates) > 0 and output_dir and hst_path:
                basename = os.path.splitext(os.path.basename(hst_path))[0]
                out_path = os.path.join(output_dir, f"{basename}_hst_sn.png")
                save_candidates_image(hst_data, hst_candidates, out_path)

    except Exception as e:
        print(f"  Error processing {obj_key}: {e}")

    return results


def parse_filename(filename):
    """Extract ID and Field from a filename.

    Examples:
        783487_B10_hstcosmos.png  -> ("783487", "B10", "hstcosmos")
        483943_B4_jwst1.png       -> ("483943", "B4", "jwst1")
        483943_B4_jwst2.png       -> ("483943", "B4", "jwst2")

    Returns:
        (object_id, field, image_type) or None if parsing fails.
    """
    basename = os.path.splitext(filename)[0]
    match = re.match(r"^(.+?)_([^_]+)_(hstcosmos|jwst[12])$", basename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None


def build_file_maps(hst_dir, jwst_dir):
    """Build mapping from object ID to HST and JWST file paths.

    Matches by object ID only (field names may differ between HST and JWST).

    Returns:
        dict: {object_id: {"hst": path, "hst_field": str,
                           "jwst1": path, "jwst2": path, "jwst_field": str}}
    """
    print("Scanning HST directory...")
    hst_files = sorted(glob.glob(os.path.join(hst_dir, "*.png")))
    print("Scanning JWST directory...")
    jwst_files = sorted(glob.glob(os.path.join(jwst_dir, "*.png")))

    print(f"Found {len(hst_files)} HST files, {len(jwst_files)} JWST files")

    print("Building file map (matching by object ID)...")
    file_map = {}

    for f in hst_files:
        parsed = parse_filename(os.path.basename(f))
        if parsed:
            obj_id, field, _ = parsed
            if obj_id not in file_map:
                file_map[obj_id] = {}
            file_map[obj_id]["hst"] = f
            file_map[obj_id]["hst_field"] = field

    for f in jwst_files:
        parsed = parse_filename(os.path.basename(f))
        if parsed:
            obj_id, field, img_type = parsed
            if obj_id not in file_map:
                file_map[obj_id] = {}
            file_map[obj_id][img_type] = f  # "jwst1" or "jwst2"
            file_map[obj_id]["jwst_field"] = field

    return file_map


def main():
    parser = argparse.ArgumentParser(
        description="Find supernova candidates by comparing JWST and HST PNG images."
    )

    parser.add_argument(
        "--hst-dir",
        default="/data/astrofs2_1/suzuki/data/HST/cosmosacs/original_png/",
        help="Directory containing HST PNG images"
    )
    parser.add_argument(
        "--jwst-dir",
        default="/data/astrofs2_1/suzuki/data/JWST/cosmosweb/v0.8_png",
        help="Directory containing JWST PNG images"
    )
    parser.add_argument(
        "--match-radius", type=float, default=10.0,
        help="Match radius in pixels (default: 10.0)"
    )
    parser.add_argument(
        "--min-brightness", type=float, default=0,
        help="Minimum brightness for candidates (default: 0)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save annotated images (optional)"
    )
    parser.add_argument(
        "--output-csv", default="supernova_candidates.csv",
        help="Output CSV file (default: supernova_candidates.csv)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    num_workers = args.num_workers or cpu_count()
    print(f"Using {num_workers} parallel workers")

    # Build file mapping by object key (ID_Field)
    file_map = build_file_maps(args.hst_dir, args.jwst_dir)

    # Filter to objects that have both HST and at least one JWST image
    paired = {k: v for k, v in file_map.items()
              if "hst" in v and ("jwst1" in v or "jwst2" in v)}

    hst_only = sum(1 for v in file_map.values()
                   if "hst" in v and "jwst1" not in v and "jwst2" not in v)
    jwst_only = sum(1 for v in file_map.values()
                    if "hst" not in v and ("jwst1" in v or "jwst2" in v))
    print(f"Matched {len(paired)} objects with both HST and JWST images")
    print(f"HST-only (skipped): {hst_only}, JWST-only (skipped): {jwst_only}")

    if len(paired) == 0:
        hst_examples = sorted(glob.glob(os.path.join(args.hst_dir, "*.png")))[:5]
        jwst_examples = sorted(glob.glob(os.path.join(args.jwst_dir, "*.png")))[:5]
        print(f"\nHST examples: {[os.path.basename(f) for f in hst_examples]}")
        print(f"JWST examples: {[os.path.basename(f) for f in jwst_examples]}")
        return

    # Prepare work items for multiprocessing
    work_items = [
        (obj_key, paired[obj_key], args.match_radius, args.min_brightness, args.output_dir)
        for obj_key in sorted(paired.keys())
    ]

    # Process in parallel
    start_time = time.time()
    all_candidates = []
    completed = 0
    total = len(work_items)

    with Pool(processes=num_workers) as pool:
        for results in pool.imap_unordered(process_object, work_items, chunksize=100):
            all_candidates.extend(results)
            completed += 1
            if completed % 1000 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"  Progress: {completed}/{total} objects "
                      f"({completed/total*100:.1f}%) "
                      f"| {rate:.0f} obj/s | ETA: {eta:.0f}s "
                      f"| Candidates so far: {len(all_candidates)}")

    elapsed = time.time() - start_time
    print(f"\nProcessing complete in {elapsed:.1f}s")

    # Save all candidates to CSV
    if all_candidates:
        # Sort by brightness (brightest first)
        all_candidates.sort(key=lambda r: r[6], reverse=True)

        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["object_id", "hst_field", "jwst_field", "detection_type",
                             "x", "y", "brightness"])
            writer.writerows(all_candidates)
        print(f"All candidates saved to {args.output_csv}")
        print(f"Total supernova candidates: {len(all_candidates)}")
    else:
        print("\nNo supernova candidates found.")

    # Validation against known supernovae
    detected_ids = set(r[0] for r in all_candidates)
    recovered = KNOWN_SUPERNOVAE_SET & detected_ids
    missed = KNOWN_SUPERNOVAE_SET - detected_ids
    false_positives = detected_ids - KNOWN_SUPERNOVAE_SET

    print(f"\n--- Validation against {len(KNOWN_SUPERNOVAE_SET)} known supernovae ---")
    print(f"Recovered: {len(recovered)}/{len(KNOWN_SUPERNOVAE_SET)} "
          f"({len(recovered)/len(KNOWN_SUPERNOVAE_SET)*100:.1f}%)")
    if recovered:
        print(f"  IDs: {sorted(recovered)}")
    print(f"Missed: {len(missed)}/{len(KNOWN_SUPERNOVAE_SET)}")
    if missed:
        print(f"  IDs: {sorted(missed)}")
    print(f"New candidates (not in known list): {len(false_positives)}")


if __name__ == "__main__":
    main()
