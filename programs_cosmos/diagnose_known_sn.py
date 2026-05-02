"""
Diagnostic: visualize the known supernovae in the verified ID range (1-53131)
to understand what the CNN is actually seeing.

Generates side-by-side images of HST, JWST1, JWST2 for each known SN,
plus the difference images, so we can see what works and what doesn't.

The list of known SN is loaded from data/sn_v03.csv (ID, TELESCOPE).

Usage:
    python diagnose_known_sn.py
"""

import csv
import glob
import os
import re

import cv2
import numpy as np


IMAGE_SIZE = 256  # Large for visual inspection

# Verified-by-visual-inspection ID range
KNOWN_SN_RANGE_MAX = 53131

# Default path to the ground-truth supernova catalog (relative to this script).
DEFAULT_KNOWN_SN_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "sn_v03.csv"
)

HST_DIR = "/data/astrofs2_1/suzuki/data/HST/cosmosacs/original_png/"
JWST_DIR = "/data/astrofs2_1/suzuki/data/JWST/cosmosweb/v0.8_png"
OUTPUT_DIR = "diagnose_known_sn"


def load_known_sn_in_range(csv_path, max_id):
    """Load known SN IDs <= max_id from the catalog CSV (returns list)."""
    ids = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj_id = (row.get("ID") or "").strip()
            if obj_id.isdigit() and int(obj_id) <= max_id:
                ids.append(obj_id)
    return ids


def parse_filename(filename):
    basename = os.path.splitext(filename)[0]
    match = re.match(r"^(.+?)_([^_]+)_(hstcosmos|jwst[12])$", basename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None


def build_file_maps(hst_dir, jwst_dir):
    file_map = {}
    for f in sorted(glob.glob(os.path.join(hst_dir, "*.png"))):
        parsed = parse_filename(os.path.basename(f))
        if parsed:
            obj_id, field, _ = parsed
            if obj_id not in file_map:
                file_map[obj_id] = {}
            file_map[obj_id]["hst"] = f
            file_map[obj_id]["hst_field"] = field
    for f in sorted(glob.glob(os.path.join(jwst_dir, "*.png"))):
        parsed = parse_filename(os.path.basename(f))
        if parsed:
            obj_id, field, img_type = parsed
            if obj_id not in file_map:
                file_map[obj_id] = {}
            file_map[obj_id][img_type] = f
            file_map[obj_id]["jwst_field"] = field
    return file_map


def load_gray(filepath, size=IMAGE_SIZE):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def to_color(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def add_label(img, text, pos=(10, 25), color=(0, 255, 0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    KNOWN_SN_RANGE1 = load_known_sn_in_range(DEFAULT_KNOWN_SN_CSV, KNOWN_SN_RANGE_MAX)
    known_sn_set = set(KNOWN_SN_RANGE1)
    print(f"Loaded {len(KNOWN_SN_RANGE1)} known SN with ID <= "
          f"{KNOWN_SN_RANGE_MAX} from {DEFAULT_KNOWN_SN_CSV}")

    print("Scanning files...")
    file_map = build_file_maps(HST_DIR, JWST_DIR)

    # Also pick some non-SN objects nearby for comparison
    non_sn_examples = []
    for obj_id in sorted(file_map.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        if (obj_id not in known_sn_set
                and obj_id.isdigit()
                and int(obj_id) <= KNOWN_SN_RANGE_MAX):
            files = file_map[obj_id]
            if "hst" in files and "jwst1" in files and "jwst2" in files:
                non_sn_examples.append(obj_id)
                if len(non_sn_examples) >= 10:
                    break

    all_ids = KNOWN_SN_RANGE1 + non_sn_examples
    labels_map = {oid: "SN" for oid in KNOWN_SN_RANGE1}
    for oid in non_sn_examples:
        labels_map[oid] = "NOT_SN"

    for obj_id in all_ids:
        if obj_id not in file_map:
            print(f"  {obj_id}: NOT FOUND in file_map")
            continue

        files = file_map[obj_id]
        has_hst = "hst" in files
        has_jwst1 = "jwst1" in files
        has_jwst2 = "jwst2" in files

        label = labels_map.get(obj_id, "?")
        print(f"\n  {obj_id} [{label}]: HST={'Y' if has_hst else 'N'} "
              f"JWST1={'Y' if has_jwst1 else 'N'} "
              f"JWST2={'Y' if has_jwst2 else 'N'}")

        if has_hst:
            print(f"    HST: {os.path.basename(files['hst'])}")
            orig = cv2.imread(files['hst'], cv2.IMREAD_GRAYSCALE)
            if orig is not None:
                print(f"    HST original size: {orig.shape}")
        if has_jwst1:
            print(f"    JWST1: {os.path.basename(files['jwst1'])}")
            orig = cv2.imread(files['jwst1'], cv2.IMREAD_GRAYSCALE)
            if orig is not None:
                print(f"    JWST1 original size: {orig.shape}")
        if has_jwst2:
            print(f"    JWST2: {os.path.basename(files['jwst2'])}")
            orig = cv2.imread(files['jwst2'], cv2.IMREAD_GRAYSCALE)
            if orig is not None:
                print(f"    JWST2 original size: {orig.shape}")

        if not (has_hst and has_jwst1 and has_jwst2):
            print(f"    SKIPPING: missing images")
            continue

        hst = load_gray(files["hst"])
        jwst1 = load_gray(files["jwst1"])
        jwst2 = load_gray(files["jwst2"])

        if hst is None or jwst1 is None or jwst2 is None:
            print(f"    SKIPPING: failed to load")
            continue

        # Row 1: Original images
        hst_c = to_color(hst)
        jwst1_c = to_color(jwst1)
        jwst2_c = to_color(jwst2)

        add_label(hst_c, f"HST  [{label}] ID={obj_id}")
        add_label(jwst1_c, f"JWST1 [{label}] ID={obj_id}")
        add_label(jwst2_c, f"JWST2 [{label}] ID={obj_id}")

        row1 = np.hstack([hst_c, jwst1_c, jwst2_c])

        # Row 2: What the CNN sees at 64x64
        hst_small = cv2.resize(hst, (64, 64), interpolation=cv2.INTER_AREA)
        jwst1_small = cv2.resize(jwst1, (64, 64), interpolation=cv2.INTER_AREA)
        jwst2_small = cv2.resize(jwst2, (64, 64), interpolation=cv2.INTER_AREA)

        # Upscale back for display
        hst_up = to_color(cv2.resize(hst_small, (IMAGE_SIZE, IMAGE_SIZE),
                                      interpolation=cv2.INTER_NEAREST))
        jwst1_up = to_color(cv2.resize(jwst1_small, (IMAGE_SIZE, IMAGE_SIZE),
                                        interpolation=cv2.INTER_NEAREST))
        jwst2_up = to_color(cv2.resize(jwst2_small, (IMAGE_SIZE, IMAGE_SIZE),
                                        interpolation=cv2.INTER_NEAREST))

        add_label(hst_up, "HST 64x64", color=(0, 255, 255))
        add_label(jwst1_up, "JWST1 64x64", color=(0, 255, 255))
        add_label(jwst2_up, "JWST2 64x64", color=(0, 255, 255))

        row2 = np.hstack([hst_up, jwst1_up, jwst2_up])

        # Row 3: Center-cropped 50% at 64x64 (what CNN actually uses)
        def crop_center_resize(img, crop_frac=0.5):
            h, w = img.shape
            ch, cw = int(h * crop_frac), int(w * crop_frac)
            y0 = (h - ch) // 2
            x0 = (w - cw) // 2
            return cv2.resize(img[y0:y0+ch, x0:x0+cw], (64, 64),
                             interpolation=cv2.INTER_AREA)

        hst_crop = crop_center_resize(hst)
        jwst1_crop = crop_center_resize(jwst1)
        jwst2_crop = crop_center_resize(jwst2)

        hst_crop_up = to_color(cv2.resize(hst_crop, (IMAGE_SIZE, IMAGE_SIZE),
                                           interpolation=cv2.INTER_NEAREST))
        jwst1_crop_up = to_color(cv2.resize(jwst1_crop, (IMAGE_SIZE, IMAGE_SIZE),
                                              interpolation=cv2.INTER_NEAREST))
        jwst2_crop_up = to_color(cv2.resize(jwst2_crop, (IMAGE_SIZE, IMAGE_SIZE),
                                              interpolation=cv2.INTER_NEAREST))

        add_label(hst_crop_up, "HST cropped+64x64", color=(0, 128, 255))
        add_label(jwst1_crop_up, "JWST1 cropped+64x64", color=(0, 128, 255))
        add_label(jwst2_crop_up, "JWST2 cropped+64x64", color=(0, 128, 255))

        row3 = np.hstack([hst_crop_up, jwst1_crop_up, jwst2_crop_up])

        # Combine all rows
        panel = np.vstack([row1, row2, row3])

        tag = "SN" if obj_id in known_sn_set else "NEG"
        out_path = os.path.join(OUTPUT_DIR, f"{tag}_{obj_id}.png")
        cv2.imwrite(out_path, panel)
        print(f"    Saved: {out_path}")

    print(f"\nDone! Check images in: {OUTPUT_DIR}/")
    print(f"Compare SN_*.png (supernovae) vs NEG_*.png (not supernovae)")
    print(f"Can you see the SN by eye in the 64x64 cropped images?")


if __name__ == "__main__":
    main()
