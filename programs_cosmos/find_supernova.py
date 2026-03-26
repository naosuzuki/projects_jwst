"""
Find Supernova by Comparing JWST and HST PNG Images

Detects transient bright sources (supernova candidates) by aligning and
subtracting an HST reference image from a JWST observation. Since the input
images are PNGs (no WCS metadata), alignment is performed using feature
matching (ORB) and homography estimation.

Usage:
    python find_supernova.py <hst_image> <jwst_image> [--threshold SIGMA] [--output OUTPUT]
"""

import argparse

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder


def load_image(filepath):
    """Load a PNG image and convert to grayscale float64."""
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {filepath}")
    # Convert to grayscale if color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return gray.astype(np.float64)


def align_images(reference, target, max_features=5000):
    """Align target image to reference using ORB feature matching + homography.

    Args:
        reference: Grayscale reference image (HST).
        target: Grayscale target image (JWST) to be aligned.
        max_features: Maximum number of ORB features to detect.

    Returns:
        Aligned target image warped to match the reference pixel grid.
    """
    ref_8bit = cv2.normalize(reference, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    tgt_8bit = cv2.normalize(target, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(ref_8bit, None)
    kp2, des2 = orb.detectAndCompute(tgt_8bit, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise RuntimeError("Not enough features detected for alignment.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    if len(matches) < 4:
        raise RuntimeError(f"Not enough matches for alignment: {len(matches)} found, 4 required.")

    pts_ref = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_tgt = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts_tgt, pts_ref, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Homography estimation failed.")

    inliers = mask.ravel().sum()
    print(f"Alignment: {len(matches)} matches, {inliers} inliers")

    h, w = reference.shape
    aligned = cv2.warpPerspective(target, H, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return aligned


def match_psf(data, source_fwhm_pix, target_fwhm_pix):
    """Convolve the sharper image to match the broader PSF."""
    if source_fwhm_pix >= target_fwhm_pix:
        return data
    kernel_fwhm = np.sqrt(target_fwhm_pix**2 - source_fwhm_pix**2)
    kernel_sigma = kernel_fwhm / 2.355
    return gaussian_filter(data, sigma=kernel_sigma)


def detect_candidates(diff_image, threshold_sigma=5.0, fwhm=3.0):
    """Detect point-source candidates in the difference image."""
    valid = np.isfinite(diff_image) & (diff_image != 0)
    if not np.any(valid):
        return None
    mean, median, std = sigma_clipped_stats(diff_image[valid], sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    clean_diff = np.where(valid, diff_image - median, 0.0)
    sources = daofind(clean_diff)
    return sources


def save_results(sources, output_path):
    """Save detected candidates to a CSV file."""
    if sources is None or len(sources) == 0:
        print("No supernova candidates detected.")
        return
    sources.write(output_path, format="csv", overwrite=True)
    print(f"Results saved to {output_path}")


def save_diff_image(diff, output_path):
    """Save the difference image as a PNG for visual inspection."""
    # Normalize to 0-255 for display
    vmin, vmax = np.nanpercentile(diff[diff != 0], [1, 99])
    clipped = np.clip(diff, vmin, vmax)
    normalized = ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    cv2.imwrite(output_path, normalized)
    print(f"Difference image saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Find supernova candidates by comparing JWST and HST PNG images."
    )
    parser.add_argument("hst_image", help="Path to the HST reference PNG image")
    parser.add_argument("jwst_image", help="Path to the JWST observation PNG image")
    parser.add_argument(
        "--threshold", type=float, default=5.0,
        help="Detection threshold in sigma (default: 5.0)"
    )
    parser.add_argument(
        "--fwhm", type=float, default=3.0,
        help="Expected source FWHM in pixels (default: 3.0)"
    )
    parser.add_argument(
        "--psf-hst", type=float, default=2.5,
        help="HST PSF FWHM in pixels (default: 2.5)"
    )
    parser.add_argument(
        "--psf-jwst", type=float, default=1.5,
        help="JWST PSF FWHM in pixels (default: 1.5)"
    )
    parser.add_argument(
        "--output", default="supernova_candidates.csv",
        help="Output CSV file path (default: supernova_candidates.csv)"
    )
    parser.add_argument(
        "--save-diff", default=None,
        help="Save difference image as PNG (optional)"
    )
    args = parser.parse_args()

    # Load both images
    print(f"Loading HST image: {args.hst_image}")
    hst_data = load_image(args.hst_image)
    print(f"  Shape: {hst_data.shape}")

    print(f"Loading JWST image: {args.jwst_image}")
    jwst_data = load_image(args.jwst_image)
    print(f"  Shape: {jwst_data.shape}")

    # Align JWST image to HST pixel grid using feature matching
    print("Aligning JWST image to HST reference...")
    jwst_aligned = align_images(hst_data, jwst_data)

    # PSF matching: convolve the sharper image to match the broader one
    print(f"PSF matching (HST FWHM={args.psf_hst} px, JWST FWHM={args.psf_jwst} px)...")
    if args.psf_jwst < args.psf_hst:
        jwst_matched = match_psf(jwst_aligned, args.psf_jwst, args.psf_hst)
        hst_matched = hst_data
        detection_fwhm = args.psf_hst
    else:
        hst_matched = match_psf(hst_data, args.psf_hst, args.psf_jwst)
        jwst_matched = jwst_aligned
        detection_fwhm = args.psf_jwst

    # Flux scaling: normalize brightness in the overlap region
    overlap = (jwst_matched > 0) & (hst_matched > 0)
    if np.sum(overlap) > 0:
        ratio = np.median(jwst_matched[overlap]) / np.median(hst_matched[overlap])
        print(f"Flux scaling ratio (JWST/HST): {ratio:.4f}")
        hst_matched = hst_matched * ratio

    # Image subtraction
    print("Subtracting images (JWST - HST)...")
    diff = jwst_matched - hst_matched
    # Zero out regions outside the overlap
    diff[~overlap] = 0

    # Optionally save the difference image
    if args.save_diff:
        save_diff_image(diff, args.save_diff)

    # Detect candidates
    print(f"Detecting candidates (threshold={args.threshold} sigma, fwhm={detection_fwhm:.1f} px)...")
    sources = detect_candidates(diff, threshold_sigma=args.threshold, fwhm=detection_fwhm)

    if sources is not None and len(sources) > 0:
        print(f"\nFound {len(sources)} supernova candidate(s):")
        print(f"{'ID':>4}  {'X':>10}  {'Y':>10}  {'Flux':>12}  {'Peak':>10}")
        print("-" * 52)
        for row in sources:
            print(
                f"{row['id']:4d}  {row['xcentroid']:10.2f}  {row['ycentroid']:10.2f}  "
                f"{row['flux']:12.2f}  {row['peak']:10.2f}"
            )
    else:
        print("\nNo supernova candidates detected.")

    save_results(sources, args.output)


if __name__ == "__main__":
    main()
