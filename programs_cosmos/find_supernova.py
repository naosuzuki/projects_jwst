"""
Find Supernova by Comparing JWST and HST PNG Images

Detects supernova candidates by finding bright sources in the JWST image
that have no counterpart in the HST image. Avoids image subtraction by
detecting sources independently in each image and cross-matching.

HST directory: /data/astrofs2_1/suzuki/data/HST/cosmosacs/original_png/
JWST directory: /data/astrofs2_1/suzuki/data/JWST/cosmosweb/v0.8_png

Usage:
    python find_supernova.py <hst_image> <jwst_image> [options]
    python find_supernova.py --hst-dir <dir> --jwst-dir <dir> [options]
"""

import argparse
import glob
import os

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


def main():
    parser = argparse.ArgumentParser(
        description="Find supernova candidates by comparing JWST and HST PNG images."
    )

    # Single pair mode
    parser.add_argument("hst_image", nargs="?", default=None,
                        help="Path to the HST reference PNG image")
    parser.add_argument("jwst_image", nargs="?", default=None,
                        help="Path to the JWST observation PNG image")

    # Batch mode
    parser.add_argument("--hst-dir", default=None,
                        help="Directory containing HST PNG images")
    parser.add_argument("--jwst-dir", default=None,
                        help="Directory containing JWST PNG images")

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

    all_candidates = []

    if args.hst_dir and args.jwst_dir:
        # Batch mode: process all matching pairs
        hst_files = sorted(glob.glob(os.path.join(args.hst_dir, "*.png")))
        jwst_files = sorted(glob.glob(os.path.join(args.jwst_dir, "*.png")))

        hst_map = {os.path.basename(f): f for f in hst_files}
        jwst_map = {os.path.basename(f): f for f in jwst_files}

        common = sorted(set(hst_map.keys()) & set(jwst_map.keys()))
        print(f"Found {len(hst_files)} HST files, {len(jwst_files)} JWST files, "
              f"{len(common)} matching pairs")

        if len(common) == 0:
            print("\nNo matching filenames found between directories.")
            print(f"HST examples: {[os.path.basename(f) for f in hst_files[:5]]}")
            print(f"JWST examples: {[os.path.basename(f) for f in jwst_files[:5]]}")
            return

        for fname in common:
            candidates = process_pair(
                hst_map[fname], jwst_map[fname],
                args.match_radius, args.min_brightness, args.output_dir
            )
            if len(candidates) > 0:
                for x, y, bright in candidates:
                    all_candidates.append([fname, x, y, bright])

    elif args.hst_image and args.jwst_image:
        # Single pair mode
        candidates = process_pair(
            args.hst_image, args.jwst_image,
            args.match_radius, args.min_brightness, args.output_dir
        )
        if len(candidates) > 0:
            fname = os.path.basename(args.jwst_image)
            for x, y, bright in candidates:
                all_candidates.append([fname, x, y, bright])
    else:
        parser.error("Provide either two image paths or --hst-dir and --jwst-dir")

    # Save all candidates to CSV
    if all_candidates:
        import csv
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "x", "y", "brightness"])
            writer.writerows(all_candidates)
        print(f"\nAll candidates saved to {args.output_csv}")
        print(f"Total supernova candidates: {len(all_candidates)}")
    else:
        print("\nNo supernova candidates found.")


if __name__ == "__main__":
    main()
