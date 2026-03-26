"""
Find Supernova by Comparing JWST and HST Images

Detects transient bright sources (supernova candidates) by aligning and
subtracting an HST reference image from a JWST observation (or vice versa).
Handles the different pixel scales, resolutions, and WCS of the two telescopes
via reprojection.

JWST NIRCam: ~0.031 arcsec/pixel (short-wave), ~0.063 arcsec/pixel (long-wave)
HST ACS/WFC: ~0.05 arcsec/pixel
HST WFC3/IR: ~0.13 arcsec/pixel

Usage:
    python find_supernova.py <hst_image> <jwst_image> [--threshold SIGMA] [--output OUTPUT]
"""

import argparse

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder
from reproject import reproject_interp


def load_fits(filepath):
    """Load a FITS image and return (data, header) from the science extension."""
    with fits.open(filepath) as hdul:
        # Try SCI extension first (standard for HST/JWST), fall back to primary
        for ext in ["SCI", 0]:
            try:
                data = hdul[ext].data
                header = hdul[ext].header
                if data is not None:
                    return data.astype(np.float64), header
            except (KeyError, IndexError):
                continue
    raise ValueError(f"No valid image data found in {filepath}")


def get_pixel_scale(header):
    """Estimate pixel scale in arcsec/pixel from the WCS header."""
    wcs = WCS(header)
    # Use the pixel scale matrix to compute scale
    pixel_scales = np.abs(wcs.wcs.cdelt) * 3600.0  # deg -> arcsec
    if np.any(pixel_scales == 0):
        # Try CD matrix instead
        cd = wcs.wcs.cd
        pixel_scales = np.sqrt(np.sum(cd**2, axis=0)) * 3600.0
    return np.mean(pixel_scales)


def reproject_to_match(data_to_reproject, header_to_reproject, target_header):
    """Reproject one image to match the WCS and pixel grid of another."""
    input_hdu = fits.PrimaryHDU(data=data_to_reproject, header=header_to_reproject)
    reprojected, footprint = reproject_interp(input_hdu, target_header)
    # Mask regions outside the footprint
    reprojected[footprint == 0] = np.nan
    return reprojected, footprint


def match_psf(data, source_fwhm_pix, target_fwhm_pix):
    """Convolve the sharper image to match the broader PSF.

    Only convolves if source_fwhm_pix < target_fwhm_pix.
    The convolution kernel size is computed to broaden the PSF
    from source_fwhm to target_fwhm in quadrature.
    """
    if source_fwhm_pix >= target_fwhm_pix:
        return data
    # kernel FWHM in quadrature: target^2 = source^2 + kernel^2
    kernel_fwhm = np.sqrt(target_fwhm_pix**2 - source_fwhm_pix**2)
    kernel_sigma = kernel_fwhm / 2.355  # FWHM to sigma
    kernel = Gaussian2DKernel(kernel_sigma)
    return convolve(data, kernel, nan_treatment="interpolate")


def subtract_images(reference, new_image):
    """Subtract reference from new image to reveal transients."""
    diff = new_image - reference
    return diff


def detect_candidates(diff_image, threshold_sigma=5.0, fwhm=3.0):
    """Detect point-source candidates in the difference image."""
    # Mask NaN pixels for statistics
    valid = np.isfinite(diff_image)
    mean, median, std = sigma_clipped_stats(diff_image[valid], sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    # Replace NaNs with median for detection
    clean_diff = np.where(valid, diff_image - median, 0.0)
    sources = daofind(clean_diff)
    return sources


def add_wcs_coordinates(sources, header):
    """Add RA/Dec columns to the source table using WCS."""
    if sources is None or len(sources) == 0:
        return sources
    wcs = WCS(header)
    ra, dec = wcs.all_pix2world(sources["xcentroid"], sources["ycentroid"], 0)
    sources["ra"] = ra
    sources["dec"] = dec
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
        description="Find supernova candidates by comparing JWST and HST FITS images."
    )
    parser.add_argument("hst_image", help="Path to the HST reference FITS image")
    parser.add_argument("jwst_image", help="Path to the JWST observation FITS image")
    parser.add_argument(
        "--threshold", type=float, default=5.0,
        help="Detection threshold in sigma (default: 5.0)"
    )
    parser.add_argument(
        "--fwhm", type=float, default=3.0,
        help="Expected source FWHM in pixels after PSF matching (default: 3.0)"
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
    args = parser.parse_args()

    # Load both images
    print(f"Loading HST image: {args.hst_image}")
    hst_data, hst_header = load_fits(args.hst_image)

    print(f"Loading JWST image: {args.jwst_image}")
    jwst_data, jwst_header = load_fits(args.jwst_image)

    # Report pixel scales
    hst_scale = get_pixel_scale(hst_header)
    jwst_scale = get_pixel_scale(jwst_header)
    print(f"HST pixel scale:  {hst_scale:.4f} arcsec/pixel")
    print(f"JWST pixel scale: {jwst_scale:.4f} arcsec/pixel")

    # Reproject HST image to match JWST pixel grid (higher resolution)
    print("Reprojecting HST image to JWST pixel grid...")
    hst_reprojected, footprint = reproject_to_match(hst_data, hst_header, jwst_header)

    # PSF matching: convolve the sharper image to match the broader one
    # After reprojection, PSF FWHMs need to be in the target (JWST) pixel scale
    hst_fwhm_jwst_pix = args.psf_hst * (hst_scale / jwst_scale)
    jwst_fwhm_pix = args.psf_jwst

    print(f"PSF FWHM on JWST grid — HST: {hst_fwhm_jwst_pix:.2f} px, JWST: {jwst_fwhm_pix:.2f} px")

    if jwst_fwhm_pix < hst_fwhm_jwst_pix:
        print("Convolving JWST image to match HST PSF...")
        jwst_matched = match_psf(jwst_data, jwst_fwhm_pix, hst_fwhm_jwst_pix)
        detection_fwhm = hst_fwhm_jwst_pix
    else:
        print("Convolving HST image to match JWST PSF...")
        hst_reprojected = match_psf(hst_reprojected, hst_fwhm_jwst_pix, jwst_fwhm_pix)
        jwst_matched = jwst_data
        detection_fwhm = jwst_fwhm_pix

    # Flux scaling: normalize by median ratio in the overlap region
    valid = np.isfinite(hst_reprojected) & np.isfinite(jwst_matched)
    valid &= (hst_reprojected > 0) & (jwst_matched > 0)
    if np.sum(valid) > 0:
        ratio = np.nanmedian(jwst_matched[valid] / hst_reprojected[valid])
        print(f"Flux scaling ratio (JWST/HST): {ratio:.4f}")
        hst_reprojected *= ratio

    # Image subtraction
    print("Subtracting images (JWST - HST)...")
    diff = subtract_images(hst_reprojected, jwst_matched)

    # Detect candidates
    print(f"Detecting candidates (threshold={args.threshold} sigma, fwhm={detection_fwhm:.1f} px)...")
    sources = detect_candidates(diff, threshold_sigma=args.threshold, fwhm=detection_fwhm)

    # Add WCS coordinates
    sources = add_wcs_coordinates(sources, jwst_header)

    if sources is not None and len(sources) > 0:
        print(f"\nFound {len(sources)} supernova candidate(s):")
        print(f"{'ID':>4}  {'X':>8}  {'Y':>8}  {'RA':>12}  {'Dec':>12}  {'Flux':>12}  {'Peak':>10}")
        print("-" * 72)
        for row in sources:
            print(
                f"{row['id']:4d}  {row['xcentroid']:8.2f}  {row['ycentroid']:8.2f}  "
                f"{row['ra']:12.6f}  {row['dec']:12.6f}  "
                f"{row['flux']:12.2f}  {row['peak']:10.2f}"
            )
    else:
        print("\nNo supernova candidates detected.")

    save_results(sources, args.output)


if __name__ == "__main__":
    main()
