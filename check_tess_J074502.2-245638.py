#!/usr/bin/env python3
"""
Check NASA TESS database for photometric data on star J074502.2-245638.

Target coordinates (J2000):
  RA  = 07h 45m 02.2s  = 116.259167 deg
  Dec = -24d 56m 38s    = -24.943889 deg

This script:
  1. Uses tess-point to determine TESS sector coverage (offline)
  2. Queries Gaia DR3 for the source ID and cross-match
  3. Queries the TESS Input Catalog (TIC) via MAST
  4. Checks for available TESS light curve products
  5. Downloads light curve data if available

Requirements:
  pip install astroquery astropy tess-point lightkurve
"""

import warnings
warnings.filterwarnings('ignore')

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

# =============================================================================
# Target definition
# =============================================================================
TARGET_NAME = "J074502.2-245638"
RA_DEG  = 116.259167   # 07h 45m 02.2s
DEC_DEG = -24.943889   # -24d 56m 38s
SEARCH_RADIUS_ARCSEC = 15

coord = SkyCoord(ra=RA_DEG*u.deg, dec=DEC_DEG*u.deg, frame='icrs')

print("=" * 70)
print(f"  TESS Data Check for {TARGET_NAME}")
print(f"  RA  = {RA_DEG:.6f} deg  (07h 45m 02.2s)")
print(f"  Dec = {DEC_DEG:.6f} deg (-24d 56m 38s)")
print("=" * 70)


# =============================================================================
# Step 1: TESS sector coverage (offline via tess-point)
# =============================================================================
print("\n[1] TESS Sector Coverage (tess-point, offline)")
print("-" * 50)
try:
    from tess_stars2px import tess_stars2px_function_entry

    result = tess_stars2px_function_entry(0, RA_DEG, DEC_DEG)
    sectors  = result[3]
    cameras  = result[4]
    ccds     = result[5]
    colpix   = result[6]
    rowpix   = result[7]
    ecl_long = result[1][0]
    ecl_lat  = result[2][0]

    print(f"  Ecliptic coords: lon={ecl_long:.2f} deg, lat={ecl_lat:.2f} deg")
    print(f"  Number of sectors with coverage: {len(sectors)}")
    print(f"  Sectors: {', '.join(str(s) for s in sectors)}")
    print(f"\n  {'Sector':>8} {'Camera':>8} {'CCD':>6} {'ColPix':>8} {'RowPix':>8}")
    print(f"  {'-'*42}")
    for i in range(len(sectors)):
        print(f"  {sectors[i]:>8d} {cameras[i]:>8d} {ccds[i]:>6d} {colpix[i]:>8.1f} {rowpix[i]:>8.1f}")
except Exception as e:
    print(f"  tess-point error: {e}")
    print("  Install with: pip install tess-point")


# =============================================================================
# Step 2: Gaia DR3 cross-match
# =============================================================================
print(f"\n[2] Gaia DR3 Cross-Match (radius = {SEARCH_RADIUS_ARCSEC} arcsec)")
print("-" * 50)
gaia_source_id = None
try:
    from astroquery.gaia import Gaia

    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    job = Gaia.cone_search_async(
        coord,
        radius=SEARCH_RADIUS_ARCSEC * u.arcsec,
    )
    gaia_results = job.get_results()

    if len(gaia_results) > 0:
        # Sort by G magnitude to get the brightest
        gaia_results.sort('phot_g_mean_mag')
        row = gaia_results[0]
        gaia_source_id = row['source_id']
        print(f"  Gaia DR3 Source ID: {gaia_source_id}")
        print(f"  RA:  {row['ra']:.6f} deg")
        print(f"  Dec: {row['dec']:.6f} deg")
        print(f"  G mag:  {row['phot_g_mean_mag']:.3f}")
        print(f"  BP mag: {row['phot_bp_mean_mag']:.3f}")
        print(f"  RP mag: {row['phot_rp_mean_mag']:.3f}")
        print(f"  Parallax: {row['parallax']:.4f} +/- {row['parallax_error']:.4f} mas")
        if row['parallax'] > 0:
            dist_pc = 1000.0 / row['parallax']
            print(f"  Distance (approx): {dist_pc:.1f} pc")
        print(f"  PM_RA:  {row['pmra']:.3f} mas/yr")
        print(f"  PM_Dec: {row['pmdec']:.3f} mas/yr")
        try:
            print(f"  Teff:  {row['teff_gspphot']:.0f} K")
            print(f"  log(g): {row['logg_gspphot']:.2f}")
        except:
            pass
        if len(gaia_results) > 1:
            print(f"\n  ({len(gaia_results)} total Gaia sources within {SEARCH_RADIUS_ARCSEC} arcsec)")
    else:
        print("  No Gaia DR3 sources found within search radius.")
except Exception as e:
    print(f"  Gaia query error: {e}")
    print("  Install with: pip install astroquery")


# =============================================================================
# Step 3: TESS Input Catalog (TIC) via MAST
# =============================================================================
print(f"\n[3] TESS Input Catalog (TIC) via MAST")
print("-" * 50)
tic_id = None
try:
    from astroquery.mast import Catalogs

    tic_results = Catalogs.query_region(
        coord,
        radius=SEARCH_RADIUS_ARCSEC * u.arcsec,
        catalog="TIC"
    )

    if len(tic_results) > 0:
        # Sort by TESS magnitude
        tic_results.sort('Tmag')
        row = tic_results[0]
        tic_id = row['ID']
        print(f"  TIC ID: {tic_id}")
        print(f"  RA:  {row['ra']:.6f} deg")
        print(f"  Dec: {row['dec']:.6f} deg")
        print(f"  TESS mag: {row['Tmag']}")
        print(f"  V mag:    {row['Vmag']}")
        print(f"  Gaia mag: {row['GAIAmag']}")
        print(f"  J mag:    {row['Jmag']}")
        print(f"  H mag:    {row['Hmag']}")
        print(f"  K mag:    {row['Kmag']}")
        print(f"  Teff:     {row['Teff']} K")
        print(f"  log(g):   {row['logg']}")
        print(f"  Radius:   {row['rad']} Rsun")
        print(f"  Mass:     {row['mass']} Msun")
        print(f"  Distance: {row['d']} pc")
        print(f"  Luminosity: {row['lum']} Lsun")
        print(f"  Gaia ID:  {row['GAIA']}")
        print(f"  2MASS ID: {row['twomass']}")
        print(f"  Object Type: {row['objType']}")
        if len(tic_results) > 1:
            print(f"\n  ({len(tic_results)} total TIC entries within {SEARCH_RADIUS_ARCSEC} arcsec)")
    else:
        print("  No TIC entries found within search radius.")
except Exception as e:
    print(f"  TIC query error: {e}")


# =============================================================================
# Step 4: Check for TESS data products at MAST
# =============================================================================
print(f"\n[4] TESS Data Products at MAST")
print("-" * 50)
try:
    from astroquery.mast import Observations

    obs_table = Observations.query_criteria(
        coordinates=coord,
        radius=SEARCH_RADIUS_ARCSEC * u.arcsec,
        obs_collection="TESS",
        dataproduct_type="timeseries"
    )

    if len(obs_table) > 0:
        print(f"  Found {len(obs_table)} TESS timeseries observation(s):")
        for i, row in enumerate(obs_table):
            print(f"    [{i+1}] {row['obs_id']}")
            print(f"         Target: {row['target_name']}")
            print(f"         Exposure: {row['t_exptime']} s")
            print(f"         Start: {row['t_min']}, End: {row['t_max']}")
            print(f"         Project: {row['project']}")

        # Get data products for the first observation
        products = Observations.get_product_list(obs_table[:5])
        lc_products = products[products['productSubGroupDescription'] == 'LC']
        if len(lc_products) > 0:
            print(f"\n  Found {len(lc_products)} light curve file(s) available for download.")
            print("  To download, uncomment the download section below.")
        else:
            tp_products = products[
                (products['productSubGroupDescription'] == 'TP') |
                (products['description'].astype(str) == 'Target pixel files')
            ]
            print(f"\n  No extracted light curves (LC) found.")
            print(f"  Target pixel files (TP) available: {len(tp_products)}")
            print("  You can extract light curves from FFI or TP data using lightkurve.")
    else:
        print("  No TESS timeseries data products found in MAST.")
        print("  However, data may still exist in Full Frame Images (FFIs).")
        print("  Use lightkurve to extract photometry from FFIs (see Step 5).")
except Exception as e:
    print(f"  MAST query error: {e}")


# =============================================================================
# Step 5: Try lightkurve to search and download
# =============================================================================
print(f"\n[5] lightkurve Search")
print("-" * 50)
try:
    import lightkurve as lk

    # Search for any TESS data (2-min cadence or FFI)
    search_result = lk.search_lightcurve(
        f"{RA_DEG} {DEC_DEG}",
        mission="TESS",
        radius=SEARCH_RADIUS_ARCSEC
    )

    if len(search_result) > 0:
        print(f"  Found {len(search_result)} light curve product(s):")
        print(search_result)

        # === UNCOMMENT BELOW TO DOWNLOAD AND PLOT ===
        # lc_collection = search_result.download_all()
        # lc = lc_collection.stitch()
        # lc.plot()
        # import matplotlib.pyplot as plt
        # plt.savefig("J074502.2-245638_tess_lightcurve.png", dpi=150)
        # print("\n  Light curve saved to J074502.2-245638_tess_lightcurve.png")
    else:
        print("  No pre-extracted light curves found.")
        print("  Searching for FFI cutouts...")

        search_ffi = lk.search_tesscut(f"{RA_DEG} {DEC_DEG}")
        if len(search_ffi) > 0:
            print(f"  Found {len(search_ffi)} FFI cutout(s) available via TESScut:")
            print(search_ffi)
            print("\n  To extract photometry from FFIs:")
            print("    tpf = search_ffi[0].download(cutout_size=15)")
            print("    lc = tpf.to_lightcurve(aperture_mask='threshold')")
            print("    lc.plot()")
        else:
            print("  No FFI cutouts found either.")
except ImportError:
    print("  lightkurve not installed. Install with: pip install lightkurve")
except Exception as e:
    print(f"  lightkurve error: {e}")


# =============================================================================
# Step 6: SIMBAD cross-identification
# =============================================================================
print(f"\n[6] SIMBAD Cross-Identification")
print("-" * 50)
try:
    from astroquery.simbad import Simbad

    simbad_result = Simbad.query_region(coord, radius=SEARCH_RADIUS_ARCSEC * u.arcsec)

    if simbad_result is not None and len(simbad_result) > 0:
        print(f"  Found {len(simbad_result)} SIMBAD object(s):")
        for row in simbad_result:
            print(f"    Main ID: {row['main_id']}")
            print(f"    RA:  {row['ra']:.6f} deg")
            print(f"    Dec: {row['dec']:.6f} deg")

        # Get all identifiers for the first match
        main_id = simbad_result[0]['main_id']
        ids_result = Simbad.query_objectids(main_id)
        if ids_result is not None:
            print(f"\n  All known identifiers for {main_id}:")
            for row in ids_result:
                print(f"    {row['id']}")
    else:
        print("  No SIMBAD objects found within search radius.")
except Exception as e:
    print(f"  SIMBAD query error: {e}")


print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  Target: {TARGET_NAME}")
print(f"  Coordinates: RA={RA_DEG:.6f}, Dec={DEC_DEG:.6f}")
print(f"  TESS sectors covering this position: {', '.join(str(s) for s in sectors)}")
if tic_id:
    print(f"  TIC ID: {tic_id}")
if gaia_source_id:
    print(f"  Gaia DR3 Source ID: {gaia_source_id}")
print(f"\n  NOTE: Even if no pre-extracted light curves exist,")
print(f"  TESS Full Frame Image (FFI) data should be available")
print(f"  for all {len(sectors)} sectors. Use lightkurve's TESScut")
print(f"  to extract custom photometry from FFIs.")
print("=" * 70)
