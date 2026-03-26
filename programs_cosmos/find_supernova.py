"""
Find Supernova by Comparing JWST and HST PNG Images

Detects supernova candidates by finding bright sources in the JWST image
that have no counterpart in the HST image. Avoids image subtraction by
detecting sources independently in each image and cross-matching.

Filename conventions:
    HST:  {ID}_{Field}_hstcosmos.png   (e.g., 783487_B10_hstcosmos.png)
    JWST: {ID}_{Field}_jwst1.png       (e.g., 783487_B10_jwst1.png)
          {ID}_{Field}_jwst2.png       (e.g., 783487_B10_jwst2.png)
    Images are matched by ID + Field name.
    Each HST image is compared against both jwst1 and jwst2 for that object.

HST directory: /data/astrofs2_1/suzuki/data/HST/cosmosacs/original_png/
JWST directory: /data/astrofs2_1/suzuki/data/JWST/cosmosweb/v0.8_png

Usage:
    python find_supernova.py [options]
"""

import argparse
import csv
import glob
import os
import re

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


def detect_sources(image, num_features=5000):
    """Detect bright point sources using blob detection.

    Returns array of (x, y, brightness) for each detected source.
    """
    # Normalize to 8-bit for detection
    img_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Blob detector tuned for star-like sources
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
        # Measure brightness in a small aperture
        y_lo = max(0, iy - 2)
        y_hi = min(image.shape[0], iy + 3)
        x_lo = max(0, ix - 2)
        x_hi = min(image.shape[1], ix + 3)
        brightness = np.mean(image[y_lo:y_hi, x_lo:x_hi])
        sources.append([x, y, brightness])

    return np.array(sources)


def find_new_sources(jwst_sources, hst_sources, match_radius=10.0):
    """Find JWST sources with no HST counterpart within match_radius pixels.

    Args:
        jwst_sources: Nx3 array (x, y, brightness) from JWST image.
        hst_sources: Mx3 array (x, y, brightness) from HST image.
        match_radius: Maximum distance in pixels to consider a match.

    Returns:
        Array of unmatched JWST sources (supernova candidates).
    """
    if len(jwst_sources) == 0:
        return np.empty((0, 3))
    if len(hst_sources) == 0:
        return jwst_sources

    hst_tree = cKDTree(hst_sources[:, :2])
    distances, _ = hst_tree.query(jwst_sources[:, :2])

    # Sources in JWST with no nearby HST counterpart
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
    print(f"Annotated image saved to {output_path}")


def process_pair(hst_path, jwst_path, match_radius, min_brightness, output_dir):
    """Process a single HST/JWST image pair."""
    basename = os.path.splitext(os.path.basename(jwst_path))[0]
    print(f"\nProcessing: {basename}")

    hst_data = load_image(hst_path)
    jwst_data = load_image(jwst_path)
    print(f"  HST shape: {hst_data.shape}, JWST shape: {jwst_data.shape}")

    hst_sources = detect_sources(hst_data)
    jwst_sources = detect_sources(jwst_data)
    print(f"  HST sources: {len(hst_sources)}, JWST sources: {len(jwst_sources)}")

    candidates = find_new_sources(jwst_sources, hst_sources, match_radius)

    # Filter by minimum brightness
    if len(candidates) > 0 and min_brightness > 0:
        candidates = candidates[candidates[:, 2] >= min_brightness]

    # Sort by brightness (brightest first)
    if len(candidates) > 0:
        candidates = candidates[candidates[:, 2].argsort()[::-1]]

    print(f"  Supernova candidates: {len(candidates)}")

    if len(candidates) > 0:
        for i, (x, y, bright) in enumerate(candidates):
            print(f"    #{i+1}: x={x:.1f}, y={y:.1f}, brightness={bright:.1f}")

        # Save annotated image
        if output_dir:
            out_path = os.path.join(output_dir, f"{basename}_candidates.png")
            save_candidates_image(jwst_data, candidates, out_path)

    return candidates


def parse_object_key(filename):
    """Extract the ID_Field key from a filename.

    Examples:
        783487_B10_hstcosmos.png  -> 783487_B10
        483943_B4_jwst1.png       -> 483943_B4
        483943_B4_jwst2.png       -> 483943_B4
    """
    basename = os.path.splitext(filename)[0]
    # Match ID_Field prefix (everything before _hstcosmos, _jwst1, or _jwst2)
    match = re.match(r"^(.+?)_(hstcosmos|jwst[12])$", basename)
    if match:
        return match.group(1)
    return None


def build_file_maps(hst_dir, jwst_dir):
    """Build mapping from object key to HST and JWST file paths.

    Returns:
        dict: {object_key: {"hst": path, "jwst1": path, "jwst2": path}}
    """
    hst_files = sorted(glob.glob(os.path.join(hst_dir, "*.png")))
    jwst_files = sorted(glob.glob(os.path.join(jwst_dir, "*.png")))

    print(f"Found {len(hst_files)} HST files, {len(jwst_files)} JWST files")

    file_map = {}

    for f in hst_files:
        key = parse_object_key(os.path.basename(f))
        if key:
            if key not in file_map:
                file_map[key] = {}
            file_map[key]["hst"] = f

    for f in jwst_files:
        basename = os.path.splitext(os.path.basename(f))[0]
        key = parse_object_key(os.path.basename(f))
        if key:
            if key not in file_map:
                file_map[key] = {}
            if basename.endswith("_jwst1"):
                file_map[key]["jwst1"] = f
            elif basename.endswith("_jwst2"):
                file_map[key]["jwst2"] = f

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
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Build file mapping by object key (ID_Field)
    file_map = build_file_maps(args.hst_dir, args.jwst_dir)

    # Filter to objects that have both HST and at least one JWST image
    paired = {k: v for k, v in file_map.items()
              if "hst" in v and ("jwst1" in v or "jwst2" in v)}
    print(f"Matched {len(paired)} objects with both HST and JWST images")

    if len(paired) == 0:
        # Show examples to help debug naming
        hst_examples = sorted(glob.glob(os.path.join(args.hst_dir, "*.png")))[:5]
        jwst_examples = sorted(glob.glob(os.path.join(args.jwst_dir, "*.png")))[:5]
        print(f"\nHST examples: {[os.path.basename(f) for f in hst_examples]}")
        print(f"JWST examples: {[os.path.basename(f) for f in jwst_examples]}")
        return

    all_candidates = []

    for obj_key in sorted(paired.keys()):
        files = paired[obj_key]
        hst_path = files["hst"]

        # Compare HST against each available JWST filter image
        for jwst_key in ["jwst1", "jwst2"]:
            if jwst_key not in files:
                continue
            jwst_path = files[jwst_key]

            candidates = process_pair(
                hst_path, jwst_path,
                args.match_radius, args.min_brightness, args.output_dir
            )
            if len(candidates) > 0:
                jwst_fname = os.path.basename(jwst_path)
                for x, y, bright in candidates:
                    all_candidates.append([obj_key, jwst_fname, x, y, bright])

    # Save all candidates to CSV
    if all_candidates:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["object_id", "jwst_file", "x", "y", "brightness"])
            writer.writerows(all_candidates)
        print(f"\nAll candidates saved to {args.output_csv}")
        print(f"Total supernova candidates: {len(all_candidates)}")
    else:
        print("\nNo supernova candidates found.")


if __name__ == "__main__":
    main()
