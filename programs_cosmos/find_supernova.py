"""
Find Supernova by Comparing Two Astronomical Images

Detects transient bright sources (supernova candidates) by subtracting
a reference image from a new observation and identifying significant
residual sources.

Usage:
    python find_supernova.py <reference_image> <new_image> [--threshold SIGMA] [--output OUTPUT]
"""

import argparse
import sys

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder


def load_image(filepath):
    """Load a FITS image and return its data array."""
    with fits.open(filepath) as hdul:
        data = hdul[0].data.astype(np.float64)
    return data


def subtract_images(reference, new_image):
    """Subtract reference from new image to reveal transients."""
    if reference.shape != new_image.shape:
        raise ValueError(
            f"Image dimensions do not match: {reference.shape} vs {new_image.shape}"
        )
    return new_image - reference


def detect_candidates(diff_image, threshold_sigma=5.0, fwhm=3.0):
    """Detect point-source candidates in the difference image.

    Args:
        diff_image: 2D array from image subtraction.
        threshold_sigma: Detection threshold in units of background sigma.
        fwhm: Expected full-width at half-maximum of sources in pixels.

    Returns:
        Astropy Table of detected sources, or None if no sources found.
    """
    mean, median, std = sigma_clipped_stats(diff_image, sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    sources = daofind(diff_image - median)
    return sources


def save_results(sources, output_path):
    """Save detected candidates to a CSV file."""
    if sources is None or len(sources) == 0:
        print("No supernova candidates detected.")
        return
    sources.write(output_path, format="csv", overwrite=True)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Find supernova candidates by comparing two FITS images."
    )
    parser.add_argument("reference", help="Path to the reference FITS image")
    parser.add_argument("new_image", help="Path to the new observation FITS image")
    parser.add_argument(
        "--threshold", type=float, default=5.0,
        help="Detection threshold in sigma (default: 5.0)"
    )
    parser.add_argument(
        "--fwhm", type=float, default=3.0,
        help="Expected source FWHM in pixels (default: 3.0)"
    )
    parser.add_argument(
        "--output", default="supernova_candidates.csv",
        help="Output CSV file path (default: supernova_candidates.csv)"
    )
    args = parser.parse_args()

    print(f"Loading reference image: {args.reference}")
    reference = load_image(args.reference)

    print(f"Loading new image: {args.new_image}")
    new_image = load_image(args.new_image)

    print("Subtracting images...")
    diff = subtract_images(reference, new_image)

    print(f"Detecting candidates (threshold={args.threshold} sigma, fwhm={args.fwhm} px)...")
    sources = detect_candidates(diff, threshold_sigma=args.threshold, fwhm=args.fwhm)

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
