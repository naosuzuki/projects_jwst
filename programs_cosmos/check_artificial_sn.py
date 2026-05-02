"""
Generate and save artificial supernova examples for visual inspection.

Creates side-by-side images showing:
- Original HST, JWST1, JWST2
- Modified images with artificial SN injected

If --profiles-csv is provided (the *_profiles.csv emitted by
train_supernova_cnn.py), amp/sigma are sampled from the measured real-SN
distribution instead of fixed uniform ranges.  Sigma is scaled from the
trainer's 64x64 image size to this script's IMAGE_SIZE for display.

Usage:
    python check_artificial_sn.py --num 10
    python check_artificial_sn.py --num 10 \
        --profiles-csv supernova_cnn_candidates_profiles.csv
"""

import argparse
import csv
import glob
import os
import random
import re

import cv2
import numpy as np


IMAGE_SIZE = 256          # Display size for visual inspection
TRAINER_IMAGE_SIZE = 64   # Image size train_supernova_cnn.py uses
SIGMA_SCALE = IMAGE_SIZE / TRAINER_IMAGE_SIZE  # Scale measured sigma to display res


def load_profiles_csv(path):
    """Load amp/sigma profiles emitted by train_supernova_cnn.py.

    Returns dict {"JWST": [...], "HST": [...]} with sigma values rescaled
    from TRAINER_IMAGE_SIZE to IMAGE_SIZE for display.
    """
    profiles = {"JWST": [], "HST": []}
    if not path or not os.path.exists(path):
        return profiles
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            tel = (row.get("telescope") or "").strip().upper()
            try:
                amp_a = float(row["amp_a"])
                sig_a = float(row["sigma_a"]) * SIGMA_SCALE
            except (KeyError, ValueError):
                continue
            if tel == "JWST":
                try:
                    amp_b = float(row["amp_b"])
                    sig_b = float(row["sigma_b"]) * SIGMA_SCALE
                except (KeyError, ValueError):
                    continue
                profiles["JWST"].append({
                    "amp1": amp_a, "sigma1": sig_a,
                    "amp2": amp_b, "sigma2": sig_b,
                })
            elif tel == "HST":
                profiles["HST"].append({"amp": amp_a, "sigma": sig_a})
    return profiles


def sample_jwst_params(profiles):
    """Return (amp1, sigma1, amp2, sigma2) sampled from real or default ranges."""
    jitter_amp = random.uniform(0.7, 1.3)
    jitter_sig = random.uniform(0.85, 1.15)
    if profiles and profiles.get("JWST"):
        p = random.choice(profiles["JWST"])
        amp1 = float(np.clip(p["amp1"] * jitter_amp, 0.05, 0.95))
        amp2 = float(np.clip(p["amp2"] * jitter_amp, 0.05, 0.95))
        sig1 = float(np.clip(p["sigma1"] * jitter_sig, 1.0, 20.0))
        sig2 = float(np.clip(p["sigma2"] * jitter_sig, 1.0, 20.0))
    else:
        amp1 = random.uniform(0.3, 0.9)
        amp2 = amp1 * random.uniform(0.7, 1.3)
        sig1 = sig2 = random.uniform(2.0, 5.0)
    return amp1, sig1, amp2, sig2


def sample_hst_params(profiles):
    """Return (amp, sigma) sampled from real or default ranges."""
    jitter_amp = random.uniform(0.7, 1.3)
    jitter_sig = random.uniform(0.85, 1.15)
    if profiles and profiles.get("HST"):
        p = random.choice(profiles["HST"])
        amp = float(np.clip(p["amp"] * jitter_amp, 0.05, 0.95))
        sig = float(np.clip(p["sigma"] * jitter_sig, 1.0, 20.0))
    else:
        amp = random.uniform(0.3, 0.9)
        sig = random.uniform(2.0, 5.0)
    return amp, sig


def parse_filename(filename):
    basename = os.path.splitext(filename)[0]
    match = re.match(r"^(.+?)_([^_]+)_(hstcosmos|jwst[12])$", basename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None


def build_file_maps(hst_dir, jwst_dir):
    hst_files = sorted(glob.glob(os.path.join(hst_dir, "*.png")))
    jwst_files = sorted(glob.glob(os.path.join(jwst_dir, "*.png")))
    file_map = {}
    for f in hst_files:
        parsed = parse_filename(os.path.basename(f))
        if parsed:
            obj_id, field, _ = parsed
            if obj_id not in file_map:
                file_map[obj_id] = {}
            file_map[obj_id]["hst"] = f
    for f in jwst_files:
        parsed = parse_filename(os.path.basename(f))
        if parsed:
            obj_id, field, img_type = parsed
            if obj_id not in file_map:
                file_map[obj_id] = {}
            file_map[obj_id][img_type] = f
    return file_map


def load_and_resize(filepath, size=IMAGE_SIZE):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def inject_supernova_at(image, x, y, peak, sigma):
    img = image.copy()
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    gaussian = peak * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return np.clip(img + gaussian, 0, 1).astype(np.float32)


def to_uint8(img):
    return (img * 255).clip(0, 255).astype(np.uint8)


def add_label(img_color, text, position=(10, 25)):
    cv2.putText(img_color, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def draw_crosshair(img_color, x, y, color=(0, 0, 255)):
    cx, cy = int(x), int(y)
    cv2.circle(img_color, (cx, cy), 15, color, 2)
    cv2.line(img_color, (cx - 20, cy), (cx + 20, cy), color, 1)
    cv2.line(img_color, (cx, cy - 20), (cx, cy + 20), color, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate artificial supernova examples for inspection."
    )
    parser.add_argument(
        "--hst-dir",
        default="/data/astrofs2_1/suzuki/data/HST/cosmosacs/original_png/",
    )
    parser.add_argument(
        "--jwst-dir",
        default="/data/astrofs2_1/suzuki/data/JWST/cosmosweb/v0.8_png",
    )
    parser.add_argument("--num", type=int, default=10,
                        help="Number of examples to generate (default: 10)")
    parser.add_argument("--output-dir", default="artificial_sn_check",
                        help="Output directory (default: artificial_sn_check)")
    parser.add_argument(
        "--profiles-csv", default=None,
        help="Optional CSV of measured real-SN amp/sigma "
             "(supernova_cnn_candidates_profiles.csv from train_supernova_cnn.py). "
             "Sigma is rescaled from 64-px training res to this script's display res."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    profiles = load_profiles_csv(args.profiles_csv) if args.profiles_csv else None
    if profiles:
        print(f"Loaded profiles from {args.profiles_csv}: "
              f"{len(profiles['JWST'])} JWST, {len(profiles['HST'])} HST")
    else:
        print("No profiles CSV provided -- using uniform amp/sigma ranges")

    print("Scanning files...")
    file_map = build_file_maps(args.hst_dir, args.jwst_dir)
    paired = {k: v for k, v in file_map.items()
              if "hst" in v and "jwst1" in v and "jwst2" in v}
    print(f"Found {len(paired)} complete triplets")

    keys = list(paired.keys())
    random.shuffle(keys)

    for i, obj_id in enumerate(keys[:args.num]):
        files = paired[obj_id]

        hst_img = load_and_resize(files["hst"])
        jwst1_img = load_and_resize(files["jwst1"])
        jwst2_img = load_and_resize(files["jwst2"])

        if hst_img is None or jwst1_img is None or jwst2_img is None:
            continue

        h, w = hst_img.shape

        # --- JWST supernova example (near center where host galaxy is) ---
        cx, cy = w // 2, h // 2
        max_offset = int(w * 0.3)
        x = cx + random.randint(-max_offset, max_offset)
        y = cy + random.randint(-max_offset, max_offset)
        x = max(3, min(w - 4, x))
        y = max(3, min(h - 4, y))
        amp1, sig1, amp2, sig2 = sample_jwst_params(profiles)

        jwst1_sn = inject_supernova_at(jwst1_img, x, y, amp1, sig1)
        jwst2_sn = inject_supernova_at(jwst2_img, x, y, amp2, sig2)

        # Build comparison image: top = original, bottom = with SN
        row_orig = np.hstack([
            cv2.cvtColor(to_uint8(hst_img), cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(to_uint8(jwst1_img), cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(to_uint8(jwst2_img), cv2.COLOR_GRAY2BGR),
        ])
        row_sn = np.hstack([
            cv2.cvtColor(to_uint8(hst_img), cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(to_uint8(jwst1_sn), cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(to_uint8(jwst2_sn), cv2.COLOR_GRAY2BGR),
        ])

        # Add labels
        add_label(row_orig, "HST (original)", (10, 25))
        add_label(row_orig, "JWST1 (original)", (w + 10, 25))
        add_label(row_orig, "JWST2 (original)", (2 * w + 10, 25))
        add_label(row_sn, "HST (no SN)", (10, 25))
        add_label(row_sn, "JWST1 + SN", (w + 10, 25))
        add_label(row_sn, "JWST2 + SN", (2 * w + 10, 25))

        # Draw crosshairs on SN position
        draw_crosshair(row_sn, w + x, y)
        draw_crosshair(row_sn, 2 * w + x, y)

        jwst_panel = np.vstack([row_orig, row_sn])

        out_path = os.path.join(args.output_dir,
                                f"{i+1:03d}_{obj_id}_jwst_sn.png")
        cv2.imwrite(out_path, jwst_panel)
        print(f"  JWST SN: {out_path} "
              f"(pos={x},{y} amp1={amp1:.2f} sig1={sig1:.1f} amp2={amp2:.2f} sig2={sig2:.1f})")

        # --- HST supernova example (near center where host galaxy is) ---
        x2 = cx + random.randint(-max_offset, max_offset)
        y2 = cy + random.randint(-max_offset, max_offset)
        x2 = max(3, min(w - 4, x2))
        y2 = max(3, min(h - 4, y2))
        amp_h, sigma_h = sample_hst_params(profiles)

        hst_sn = inject_supernova_at(hst_img, x2, y2, amp_h, sigma_h)

        row_orig2 = np.hstack([
            cv2.cvtColor(to_uint8(hst_img), cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(to_uint8(jwst1_img), cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(to_uint8(jwst2_img), cv2.COLOR_GRAY2BGR),
        ])
        row_sn2 = np.hstack([
            cv2.cvtColor(to_uint8(hst_sn), cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(to_uint8(jwst1_img), cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(to_uint8(jwst2_img), cv2.COLOR_GRAY2BGR),
        ])

        add_label(row_orig2, "HST (original)", (10, 25))
        add_label(row_orig2, "JWST1 (original)", (w + 10, 25))
        add_label(row_orig2, "JWST2 (original)", (2 * w + 10, 25))
        add_label(row_sn2, "HST + SN", (10, 25))
        add_label(row_sn2, "JWST1 (no SN)", (w + 10, 25))
        add_label(row_sn2, "JWST2 (no SN)", (2 * w + 10, 25))

        draw_crosshair(row_sn2, x2, y2)

        hst_panel = np.vstack([row_orig2, row_sn2])

        out_path = os.path.join(args.output_dir,
                                f"{i+1:03d}_{obj_id}_hst_sn.png")
        cv2.imwrite(out_path, hst_panel)
        print(f"  HST SN:  {out_path} "
              f"(pos={x2},{y2} amp={amp_h:.2f} sigma={sigma_h:.1f})")

    print(f"\nDone! Check images in: {args.output_dir}/")


if __name__ == "__main__":
    main()
