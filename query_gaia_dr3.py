#!/usr/bin/env python3
"""
Query Gaia DR3 for all available photometric data on
J074502.2-245638 (Gaia DR3 5614593699906384768, G=14.8).

Run this script on a machine with unrestricted internet access.

Requirements:
  pip install astroquery astropy GaiaXPy matplotlib numpy
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np

GAIA_ID = 5614593699906384768
TARGET_NAME = "J074502.2-245638"

print("=" * 70)
print(f"Gaia DR3 Query: {TARGET_NAME}")
print(f"Source ID: {GAIA_ID}")
print("=" * 70)

# =============================================================================
# 1. Basic source parameters
# =============================================================================
print("\n[1] Source Parameters")
print("-" * 50)
from astroquery.gaia import Gaia

query = f"""
SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmra_error,
  pmdec, pmdec_error, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
  bp_rp, phot_g_mean_flux, phot_g_mean_flux_error,
  phot_bp_mean_flux, phot_bp_mean_flux_error,
  phot_rp_mean_flux, phot_rp_mean_flux_error,
  phot_variable_flag, teff_gspphot, logg_gspphot, mh_gspphot,
  ag_gspphot, ebpminrp_gspphot, distance_gspphot,
  radial_velocity, radial_velocity_error,
  ruwe, astrometric_excess_noise,
  phot_g_n_obs, phot_bp_n_obs, phot_rp_n_obs,
  classprob_dsc_combmod_star, classprob_dsc_combmod_binary
FROM gaiadr3.gaia_source
WHERE source_id = {GAIA_ID}
"""
job = Gaia.launch_job(query)
result = job.get_results()

if len(result) > 0:
    r = result[0]
    print(f"  RA:  {r['ra']:.6f} deg")
    print(f"  Dec: {r['dec']:.6f} deg")
    print(f"  Parallax: {r['parallax']:.4f} +/- {r['parallax_error']:.4f} mas")
    if r['parallax'] > 0:
        print(f"  Distance (1/plx): {1000/r['parallax']:.1f} pc")
    print(f"  Distance (GSP-Phot): {r['distance_gspphot']:.1f} pc")
    print(f"  PM_RA: {r['pmra']:.3f} +/- {r['pmra_error']:.3f} mas/yr")
    print(f"  PM_Dec: {r['pmdec']:.3f} +/- {r['pmdec_error']:.3f} mas/yr")
    print(f"  G mag:  {r['phot_g_mean_mag']:.3f}  (N_obs: {r['phot_g_n_obs']})")
    print(f"  BP mag: {r['phot_bp_mean_mag']:.3f} (N_obs: {r['phot_bp_n_obs']})")
    print(f"  RP mag: {r['phot_rp_mean_mag']:.3f} (N_obs: {r['phot_rp_n_obs']})")
    print(f"  BP-RP:  {r['bp_rp']:.3f}")
    print(f"  Teff:   {r['teff_gspphot']:.0f} K")
    print(f"  log(g): {r['logg_gspphot']:.2f}")
    print(f"  [M/H]:  {r['mh_gspphot']:.2f}")
    print(f"  A_G:    {r['ag_gspphot']:.3f}")
    print(f"  E(BP-RP): {r['ebpminrp_gspphot']:.3f}")
    print(f"  RV: {r['radial_velocity']} +/- {r['radial_velocity_error']} km/s")
    print(f"  RUWE: {r['ruwe']:.3f}")
    print(f"  Astrometric excess noise: {r['astrometric_excess_noise']:.3f}")
    print(f"  Variability flag: {r['phot_variable_flag']}")
    print(f"  P(star): {r['classprob_dsc_combmod_star']:.3f}")
    print(f"  P(binary): {r['classprob_dsc_combmod_binary']:.3f}")
    print(f"  G flux: {r['phot_g_mean_flux']:.2f} +/- {r['phot_g_mean_flux_error']:.2f}")


# =============================================================================
# 2. Variability summary (if classified as variable)
# =============================================================================
print("\n[2] Variability Classification")
print("-" * 50)
query_var = f"""
SELECT * FROM gaiadr3.vari_summary
WHERE source_id = {GAIA_ID}
"""
try:
    job_var = Gaia.launch_job(query_var)
    result_var = job_var.get_results()
    if len(result_var) > 0:
        print("  Source IS in the variability table!")
        for col in result_var.colnames:
            print(f"  {col}: {result_var[col][0]}")
    else:
        print("  Source is NOT in the variability summary table.")
        print("  (Not classified as variable by Gaia DR3 pipeline)")
except Exception as e:
    print(f"  Query error: {e}")


# =============================================================================
# 3. Epoch photometry (light curve)
# =============================================================================
print("\n[3] Epoch Photometry (Light Curve)")
print("-" * 50)
try:
    # Method 1: Direct DataLink query
    from astroquery.gaia import GaiaClass
    retrieval_type = 'EPOCH_PHOTOMETRY'
    data_structure = 'INDIVIDUAL'
    data_release = 'Gaia DR3'

    epoch_data = Gaia.load_data(
        ids=[GAIA_ID],
        data_release=data_release,
        retrieval_type=retrieval_type,
        data_structure=data_structure,
        verbose=True
    )

    if epoch_data:
        for key, tables in epoch_data.items():
            print(f"\n  Data key: {key}")
            for table in tables:
                print(f"  Table columns: {table.colnames}")
                print(f"  Number of epochs: {len(table)}")

                # Separate by band
                for band in ['G', 'BP', 'RP']:
                    mask = table['band'] == band
                    if mask.sum() > 0:
                        t = table[mask]
                        print(f"\n  {band} band: {mask.sum()} epochs")
                        print(f"    Time range: {t['time'].min():.3f} to {t['time'].max():.3f} (BJD-2455197.5)")
                        print(f"    Mag range: {t['mag'].min():.3f} to {t['mag'].max():.3f}")
                        print(f"    Mean mag: {t['mag'].mean():.3f}")
                        print(f"    Std mag: {t['mag'].std():.4f}")

                # Save to file
                table.write(f'gaia_dr3_epoch_phot_{GAIA_ID}.csv',
                           format='csv', overwrite=True)
                print(f"\n  Saved to: gaia_dr3_epoch_phot_{GAIA_ID}.csv")
    else:
        print("  No epoch photometry available for this source.")
        print("  (Only ~11.7M sources have epoch photometry in DR3)")

except Exception as e:
    print(f"  Epoch photometry error: {e}")
    print("  Trying alternative method...")

    # Method 2: TAP query for epoch_photometry table
    try:
        query_ep = f"""
        SELECT source_id, transit_id, band, time, mag, flux, flux_error,
               flux_over_error, rejected_by_photometry, rejected_by_variability
        FROM gaiadr3.epoch_photometry
        WHERE source_id = {GAIA_ID}
        ORDER BY band, time
        """
        job_ep = Gaia.launch_job(query_ep)
        result_ep = job_ep.get_results()
        if len(result_ep) > 0:
            print(f"  Found {len(result_ep)} epoch photometry points!")
            for band in ['G', 'BP', 'RP']:
                mask = result_ep['band'] == band
                if mask.sum() > 0:
                    t = result_ep[mask]
                    print(f"  {band}: {mask.sum()} epochs, mag range [{t['mag'].min():.3f}, {t['mag'].max():.3f}]")
        else:
            print("  No epoch photometry found via TAP query either.")
    except Exception as e2:
        print(f"  TAP query also failed: {e2}")


# =============================================================================
# 4. BP/RP spectra
# =============================================================================
print("\n[4] BP/RP Low-Resolution Spectra")
print("-" * 50)
try:
    from gaiaxpy import calibrate, plot_spectra

    # Download and calibrate BP/RP spectra
    calibrated_spectra, sampling = calibrate([GAIA_ID])

    if calibrated_spectra is not None and len(calibrated_spectra) > 0:
        print(f"  BP/RP spectrum retrieved!")
        print(f"  Wavelength range: {sampling.min():.0f} - {sampling.max():.0f} nm")
        print(f"  N wavelength points: {len(sampling)}")

        # Save
        calibrated_spectra.to_csv(f'gaia_dr3_bprp_spectrum_{GAIA_ID}.csv', index=False)
        print(f"  Saved to: gaia_dr3_bprp_spectrum_{GAIA_ID}.csv")

        # Plot
        import matplotlib.pyplot as plt
        plot_spectra(calibrated_spectra, sampling,
                    title=f'{TARGET_NAME} (Gaia DR3 {GAIA_ID})')
        plt.savefig(f'gaia_dr3_bprp_spectrum_{GAIA_ID}.png', dpi=150)
        plt.close()
        print(f"  Plot saved to: gaia_dr3_bprp_spectrum_{GAIA_ID}.png")
    else:
        print("  No BP/RP spectrum available.")
except ImportError:
    print("  GaiaXPy not installed. Install with: pip install GaiaXPy")
except Exception as e:
    print(f"  BP/RP query error: {e}")


# =============================================================================
# 5. Nearby sources (contamination check)
# =============================================================================
print("\n[5] Nearby Gaia Sources (within 2 arcmin)")
print("-" * 50)
from astropy.coordinates import SkyCoord
import astropy.units as u

coord = SkyCoord(ra=116.259167*u.deg, dec=-24.943889*u.deg, frame='icrs')
job_nearby = Gaia.cone_search_async(coord, radius=2*u.arcmin)
nearby = job_nearby.get_results()
nearby.sort('phot_g_mean_mag')

print(f"  Found {len(nearby)} Gaia sources within 2 arcmin")
print(f"\n  {'Gaia ID':>22s} {'G mag':>6s} {'BP-RP':>6s} {'Dist(\")':>7s} {'Variable?':>10s}")
print(f"  {'-'*60}")

for row in nearby[:20]:
    src_coord = SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg, frame='icrs')
    sep = coord.separation(src_coord).arcsec
    is_target = " <-- TARGET" if row['source_id'] == GAIA_ID else ""
    var_flag = str(row['phot_variable_flag']) if row['phot_variable_flag'] else "---"
    print(f"  {row['source_id']:>22d} {row['phot_g_mean_mag']:>6.2f} "
          f"{row['bp_rp']:>6.2f} {sep:>7.1f} {var_flag:>10s}{is_target}")

# Save nearby sources
nearby.write(f'gaia_dr3_nearby_{GAIA_ID}.csv', format='csv', overwrite=True)
print(f"\n  Saved to: gaia_dr3_nearby_{GAIA_ID}.csv")


# =============================================================================
# 6. Photometric variability statistics
# =============================================================================
print("\n[6] Photometric Variability Statistics")
print("-" * 50)
if len(result) > 0:
    r = result[0]
    g_flux = r['phot_g_mean_flux']
    g_err = r['phot_g_mean_flux_error']
    snr = g_flux / g_err if g_err > 0 else 0
    print(f"  G-band S/N: {snr:.1f}")
    print(f"  G-band mean flux error: {g_err:.2f} e-/s")
    print(f"  Expected photometric precision: ~{100/snr:.3f}%")

    bp_flux = r['phot_bp_mean_flux']
    bp_err = r['phot_bp_mean_flux_error']
    rp_flux = r['phot_rp_mean_flux']
    rp_err = r['phot_rp_mean_flux_error']
    print(f"  BP S/N: {bp_flux/bp_err:.1f}")
    print(f"  RP S/N: {rp_flux/rp_err:.1f}")

print("\n" + "=" * 70)
print("DONE. Run this script locally with full internet access.")
print("=" * 70)
