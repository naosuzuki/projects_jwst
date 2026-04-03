#!/usr/bin/env python3
"""
Collect all spectroscopic redshifts ever taken for the ELAIS-N1 field.

ELAIS-N1 (European Large Area ISO Survey - North 1)
  Center: RA = 242.75 deg, Dec = +55.0 deg (J2000)
  Search radius: 2.0 deg (~12.6 sq deg, covers the ~9 sq deg core + margins)

Known sources of spectroscopic redshifts in ELAIS-N1:
  1. SDSS/BOSS DR18  - optical spectroscopy (~600+ spec-z, includes LOFAR ancillary)
  2. HETDEX-LOFAR    - Debski et al. 2024, ApJ 978, 101 (~9,710 spec-z)
                       Zenodo: https://zenodo.org/records/14194635
  3. DESI EDR/DR1    - Dark Energy Spectroscopic Instrument (partial coverage)
                       NOIRLab DataLab TAP: https://datalab.noirlab.edu/tap
  4. LoTSS Deep DR1  - Duncan et al. 2021, A&A 648, A4 (photo-z + spec-z compilation)
                       Kondapally et al. 2021, A&A 648, A3 (host-galaxy cross-match)
                       CDS: J/A+A/648/A4, LOFAR surveys website
  5. SWIRE ELAIS-N1  - Gonzalez-Solares et al. 2011, MNRAS 405, 2243 (289 spec-z)
                       VizieR: J/MNRAS/405/2243
  6. Rowan-Robinson  - Rowan-Robinson et al. 2013, MNRAS 428, 1958 (SWIRE photo-z + spec-z)
                       VizieR: J/MNRAS/428/1958
  7. HELP            - Shirley et al. 2021, MNRAS 507, 129
                       HeDaM: hedam.lam.fr/HELP/dataproducts/dmu23/dmu23_ELAIS-N1/
                       Surveys: Berta+2007, SDSS, Trichas+2010, Swinbank+2007,
                       Rowan-Robinson+2013 (WIYN/Keck/Gemini + NED), UZC, Lacy+2013
  8. NED             - NASA/IPAC Extragalactic Database (heterogeneous compilation)
  9. Arnaudova+ 2025 - LOFAR Deep Fields spectroscopic classifications
                       MNRAS 542, 2245 (includes DESI re-analysis)
 10. LoTSS Deep DR2  - Sabater et al. 2025, A&A 695, A80
 11. HETDEX HDR       - HETDEX Public Source Catalog (Lyman-alpha emitters)
                       Zenodo: https://zenodo.org/records/7448504

Usage:
  python 00_collect_specz.py

Requires: astroquery, astropy, pandas, requests
  pip install astroquery astropy pandas requests
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

warnings.filterwarnings('ignore')

# ── Field parameters ──────────────────────────────────────────────────────────
RA_CENTER = 242.75    # deg
DEC_CENTER = 55.0     # deg
SEARCH_RADIUS = 2.0   # deg
RA_MIN = RA_CENTER - SEARCH_RADIUS / np.cos(np.radians(DEC_CENTER))
RA_MAX = RA_CENTER + SEARCH_RADIUS / np.cos(np.radians(DEC_CENTER))
DEC_MIN = DEC_CENTER - SEARCH_RADIUS
DEC_MAX = DEC_CENTER + SEARCH_RADIUS

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(OUTDIR, exist_ok=True)


def _save_table(table, name):
    """Save an astropy Table or pandas DataFrame to CSV."""
    outpath = os.path.join(OUTDIR, f"{name}.csv")
    if isinstance(table, Table):
        table.to_pandas().to_csv(outpath, index=False)
    elif isinstance(table, pd.DataFrame):
        table.to_csv(outpath, index=False)
    else:
        raise ValueError(f"Unknown table type: {type(table)}")
    print(f"  -> Saved {outpath} ({len(table)} rows)")
    return outpath, len(table)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SDSS / BOSS DR18
# ══════════════════════════════════════════════════════════════════════════════
def query_sdss():
    from astroquery.sdss import SDSS

    print("\n[1/8] SDSS/BOSS DR18 spectroscopy")
    print("-" * 50)

    query = f"""
    SELECT
        s.specobjid, s.plate, s.mjd, s.fiberid,
        s.ra, s.dec, s.z AS redshift, s.zerr AS redshift_err,
        s.zwarning, s.class, s.subclass,
        s.survey, s.programname
    FROM SpecObj AS s
    WHERE s.ra BETWEEN {RA_MIN} AND {RA_MAX}
      AND s.dec BETWEEN {DEC_MIN} AND {DEC_MAX}
      AND s.zwarning = 0
      AND s.z > 0 AND s.z < 7
    """
    try:
        result = SDSS.query_sql(query, timeout=300)
        if result is not None and len(result) > 0:
            print(f"  SDSS: {len(result)} spectroscopic redshifts")
            return _save_table(result, "sdss_specz")
        print("  SDSS: no results returned")
    except Exception as e:
        print(f"  SDSS error: {e}")
    return None, 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. HETDEX-LOFAR (Debski et al. 2024) from Zenodo
# ══════════════════════════════════════════════════════════════════════════════
def download_hetdex_lofar():
    import requests

    print("\n[2/8] HETDEX-LOFAR catalog (Debski et al. 2024)")
    print("-" * 50)

    zenodo_api = "https://zenodo.org/api/records/14194635"
    try:
        resp = requests.get(zenodo_api, timeout=30)
        resp.raise_for_status()
        record = resp.json()
        files = record.get('files', [])
        print(f"  Zenodo record has {len(files)} files")

        downloaded = []
        for f in files:
            fname = f['key']
            fpath = os.path.join(OUTDIR, fname)
            if os.path.exists(fpath):
                print(f"  {fname} already downloaded")
                downloaded.append(fpath)
                continue
            dl_url = f['links']['self']
            print(f"  Downloading {fname} ({f['size']/1e6:.1f} MB)...")
            r = requests.get(dl_url, timeout=600, stream=True)
            r.raise_for_status()
            with open(fpath, 'wb') as out:
                for chunk in r.iter_content(8192):
                    out.write(chunk)
            print(f"  -> Saved {fpath}")
            downloaded.append(fpath)
        return downloaded
    except Exception as e:
        print(f"  HETDEX-LOFAR error: {e}")
        print("  Manual download: https://zenodo.org/records/14194635")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# 3. DESI EDR / DR1 via NOIRLab DataLab TAP
# ══════════════════════════════════════════════════════════════════════════════
def query_desi():
    print("\n[3/8] DESI EDR + DR1 via NOIRLab DataLab")
    print("-" * 50)

    try:
        from astroquery.utils.tap.core import TapPlus
        tap = TapPlus(url="https://datalab.noirlab.edu/tap")

        for table_name, label in [("desi_edr.zpix", "DESI EDR"),
                                   ("desi_dr1.zpix", "DESI DR1")]:
            query = f"""
            SELECT targetid, survey, program,
                   target_ra AS ra, target_dec AS dec,
                   z AS redshift, zerr AS redshift_err,
                   spectype, zwarn
            FROM {table_name}
            WHERE target_ra BETWEEN {RA_MIN} AND {RA_MAX}
              AND target_dec BETWEEN {DEC_MIN} AND {DEC_MAX}
              AND zwarn = 0 AND z > 0 AND z < 7
            """
            try:
                print(f"  Querying {label}...")
                job = tap.launch_job(query, verbose=False)
                result = job.get_results()
                if result is not None and len(result) > 0:
                    print(f"  {label}: {len(result)} redshifts")
                    return _save_table(result, f"desi_specz")
                print(f"  {label}: no results")
            except Exception as e:
                print(f"  {label} error: {e}")
    except Exception as e:
        print(f"  DESI TAP error: {e}")
    return None, 0


# ══════════════════════════════════════════════════════════════════════════════
# 4. VizieR catalogs (multiple)
# ══════════════════════════════════════════════════════════════════════════════
def query_vizier():
    from astroquery.vizier import Vizier

    print("\n[4/8] VizieR catalogs")
    print("-" * 50)

    coord = SkyCoord(ra=RA_CENTER, dec=DEC_CENTER, unit='deg', frame='icrs')
    radius = SEARCH_RADIUS * u.deg

    catalogs = {
        "J/MNRAS/405/2243":  "Gonzalez-Solares+2011 SWIRE ELAIS-N1 spec-z",
        "J/MNRAS/428/1958":  "Rowan-Robinson+2013 SWIRE photo-z (has spec-z col)",
        "J/A+A/648/A4":      "Duncan+2021 LoTSS Deep photo-z (has z_spec col)",
        "J/A+A/648/A3":      "Kondapally+2021 LoTSS Deep host-galaxy IDs",
        "J/MNRAS/523/1729":  "Kondapally+2023 LoTSS Deep classifications",
        "J/MNRAS/507/129":   "Shirley+2021 HELP",
        "J/A+A/620/A50":     "Malek+2018 HELP ELAIS-N1",
        "V/154":             "SDSS DR16 spectroscopic (VizieR mirror)",
        "VII/298":           "SDSS quasar catalog DR16Q",
        "J/MNRAS/542/2245":  "Arnaudova+2025 LOFAR spec classifications",
    }

    results = []
    for cat_id, description in catalogs.items():
        try:
            v = Vizier(columns=['**'], row_limit=100000, timeout=120)
            tables = v.query_region(coord, radius=radius, catalog=cat_id)
            if tables:
                for t in tables:
                    n = len(t)
                    tname = t.meta.get('name', cat_id)
                    print(f"  {description}: {tname} -> {n} rows")
                    safe = cat_id.replace('/', '_')
                    path, nrows = _save_table(t, f"vizier_{safe}")
                    results.append((cat_id, description, path, nrows))
            else:
                print(f"  {description}: no data returned")
        except Exception as e:
            print(f"  {description}: {e}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. NED (NASA/IPAC Extragalactic Database)
# ══════════════════════════════════════════════════════════════════════════════
def query_ned():
    from astroquery.ipac.ned import Ned

    print("\n[5/8] NED (NASA/IPAC Extragalactic Database)")
    print("-" * 50)

    coord = SkyCoord(ra=RA_CENTER, dec=DEC_CENTER, unit='deg', frame='icrs')
    try:
        result = Ned.query_region(coord, radius=SEARCH_RADIUS * u.deg,
                                   equinox='J2000.0')
        if result is not None:
            if hasattr(result['Redshift'], 'mask'):
                mask = ~result['Redshift'].mask
            else:
                mask = np.isfinite(result['Redshift'])
            result_z = result[mask]
            print(f"  NED: {len(result_z)} objects with redshifts (of {len(result)} total)")
            return _save_table(result_z, "ned_specz")
        print("  NED: no results")
    except Exception as e:
        print(f"  NED error: {e}")
    return None, 0


# ══════════════════════════════════════════════════════════════════════════════
# 6. LoTSS Deep Fields optical/IR catalog (from LOFAR surveys website)
# ══════════════════════════════════════════════════════════════════════════════
def download_lofar_deepfields():
    import requests

    print("\n[6/8] LoTSS Deep Fields ELAIS-N1 catalogs")
    print("-" * 50)
    print("  Source: https://lofar-surveys.org/deepfields_public_en1.html")

    urls = [
        # DR1 optical/Spitzer merged value-added catalog
        ("EN1_opt_spitzer_merged_vac_opt3as_irac4as_all_hpx_forpub.fits",
         "https://lofar-surveys.org/public/EN1_opt_spitzer_merged_vac_opt3as_irac4as_all_hpx_forpub.fits"),
        # DR1 radio-optical cross-match catalog
        ("en1_final_cross_match_catalogue-v1.0.fits",
         "https://lofar-surveys.org/public/ELAIS-N1/en1_final_cross_match_catalogue-v1.0.fits"),
        # Das et al. 2024 source classifications
        ("sdas_lotss_en1_pros_classifications.fits",
         "https://lofar-surveys.org/public/deepfields/data_release/en1/sdas_lotss_en1_pros_classifications.fits"),
    ]

    downloaded = []
    for fname, url in urls:
        fpath = os.path.join(OUTDIR, fname)
        if os.path.exists(fpath):
            print(f"  {fname} already exists")
            downloaded.append(fpath)
            continue
        try:
            print(f"  Downloading {fname}...")
            r = requests.get(url, timeout=600, stream=True)
            if r.status_code == 200:
                with open(fpath, 'wb') as out:
                    for chunk in r.iter_content(8192):
                        out.write(chunk)
                size_mb = os.path.getsize(fpath) / 1e6
                print(f"  -> Saved {fname} ({size_mb:.1f} MB)")
                downloaded.append(fpath)
            else:
                print(f"  {fname}: HTTP {r.status_code}")
        except Exception as e:
            print(f"  {fname}: {e}")

    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
# 7. HELP spectroscopic redshift catalog from HeDaM
# ══════════════════════════════════════════════════════════════════════════════
def download_help_specz():
    import requests

    print("\n[7/10] HELP spectroscopic redshift catalog (HeDaM)")
    print("-" * 50)
    print("  Source: hedam.lam.fr/HELP/dataproducts/dmu23/dmu23_ELAIS-N1/")
    print("  Surveys: Berta+2007, SDSS, Trichas+2010, Swinbank+2007,")
    print("           Rowan-Robinson+2013, UZC, Lacy+2013")

    urls = [
        ("ELAIS-N1-specz-v2_hedam.csv",
         "https://hedam.lam.fr/HELP/dataproducts/dmu23/dmu23_ELAIS-N1/ELAIS-N1-specz-v2_hedam.csv"),
        ("ELAIS-N1-specz-v2.csv",
         "https://hedam.lam.fr/HELP/dataproducts/dmu23/dmu23_ELAIS-N1/ELAIS-N1-specz-v2.csv"),
    ]

    for fname, url in urls:
        fpath = os.path.join(OUTDIR, fname)
        if os.path.exists(fpath):
            print(f"  {fname} already exists")
            return fpath
        try:
            print(f"  Downloading {fname}...")
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                with open(fpath, 'wb') as out:
                    out.write(r.content)
                print(f"  -> Saved {fname} ({len(r.content)/1024:.0f} KB)")
                return fpath
            else:
                print(f"  {fname}: HTTP {r.status_code}")
        except Exception as e:
            print(f"  {fname}: {e}")
    print("  Manual download: https://hedam.lam.fr/HELP/dataproducts/dmu23/dmu23_ELAIS-N1/")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 8. VizieR TAP (alternative to standard VizieR if blocked)
# ══════════════════════════════════════════════════════════════════════════════
def query_vizier_tap():
    print("\n[8/10] VizieR TAP (alternative access)")
    print("-" * 50)

    try:
        from astroquery.utils.tap.core import TapPlus

        tap = TapPlus(url="https://tapvizier.cds.unistra.fr/TAPVizieR/tap")

        # Duncan+2021 photo-z catalog with z_spec column
        query = f"""
        SELECT *
        FROM "J/A+A/648/A4/en1"
        WHERE 1=CONTAINS(POINT('ICRS', RAJ2000, DEJ2000),
                         CIRCLE('ICRS', {RA_CENTER}, {DEC_CENTER}, {SEARCH_RADIUS}))
        """
        try:
            print("  Querying Duncan+2021 (J/A+A/648/A4) via TAP...")
            job = tap.launch_job(query, verbose=False)
            result = job.get_results()
            if result is not None and len(result) > 0:
                print(f"  Duncan+2021: {len(result)} rows")
                return _save_table(result, "vizier_tap_duncan2021")
        except Exception as e:
            print(f"  VizieR TAP error: {e}")

    except Exception as e:
        print(f"  TAP import error: {e}")
    return None, 0


# ══════════════════════════════════════════════════════════════════════════════
# 8. HETDEX Public Source Catalog (HDR)
# ══════════════════════════════════════════════════════════════════════════════
def download_hetdex_public():
    import requests

    print("\n[9/10] HETDEX Public Source Catalog")
    print("-" * 50)
    print("  https://hetdex.org/data-results/")
    print("  Zenodo: https://zenodo.org/records/7448504")

    try:
        resp = requests.get("https://zenodo.org/api/records/7448504", timeout=30)
        resp.raise_for_status()
        record = resp.json()
        files = record.get('files', [])
        for f in files:
            fname = f['key']
            if 'catalog' in fname.lower() or fname.endswith('.fits'):
                fpath = os.path.join(OUTDIR, fname)
                if os.path.exists(fpath):
                    print(f"  {fname} already exists")
                    continue
                dl_url = f['links']['self']
                print(f"  Downloading {fname} ({f['size']/1e6:.1f} MB)...")
                r = requests.get(dl_url, timeout=600, stream=True)
                r.raise_for_status()
                with open(fpath, 'wb') as out:
                    for chunk in r.iter_content(8192):
                        out.write(chunk)
                print(f"  -> Saved {fpath}")
    except Exception as e:
        print(f"  HETDEX public catalog error: {e}")
        print("  Manual download: https://zenodo.org/records/7448504")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("SPECTROSCOPIC REDSHIFT COLLECTION FOR ELAIS-N1")
    print("=" * 60)
    print(f"Field center : RA = {RA_CENTER} deg, Dec = {DEC_CENTER} deg")
    print(f"Search box   : RA [{RA_MIN:.2f}, {RA_MAX:.2f}], Dec [{DEC_MIN:.1f}, {DEC_MAX:.1f}]")
    print(f"Output dir   : {OUTDIR}")

    t0 = time.time()
    summary = []

    # 1. SDSS
    path, n = query_sdss()
    if path:
        summary.append(("SDSS/BOSS DR18", n, path))

    # 2. HETDEX-LOFAR (Debski+2024)
    hetdex_files = download_hetdex_lofar()
    for f in hetdex_files:
        summary.append(("HETDEX-LOFAR Debski+2024", "FITS", f))

    # 3. DESI
    path, n = query_desi()
    if path:
        summary.append(("DESI EDR/DR1", n, path))

    # 4. VizieR catalogs
    vizier_results = query_vizier()
    for cat_id, desc, path, n in vizier_results:
        summary.append((desc, n, path))

    # 5. NED
    path, n = query_ned()
    if path:
        summary.append(("NED", n, path))

    # 6. LOFAR Deep Fields catalogs
    lofar_files = download_lofar_deepfields()
    for f in lofar_files:
        summary.append(("LoTSS Deep Fields", "FITS", f))

    # 7. HELP spec-z from HeDaM
    help_path = download_help_specz()
    if help_path:
        summary.append(("HELP dmu23", "CSV", help_path))

    # 8. VizieR TAP alternative
    path, n = query_vizier_tap()
    if path:
        summary.append(("VizieR TAP Duncan+2021", n, path))

    # 9. HETDEX public catalog
    download_hetdex_public()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"COLLECTION COMPLETE ({elapsed:.0f}s)")
    print("=" * 60)
    for desc, n, path in summary:
        print(f"  {desc}: {n} rows -> {os.path.basename(path)}")

    # Save summary
    summary_df = pd.DataFrame(summary, columns=['source', 'n_rows', 'file'])
    summary_df.to_csv(os.path.join(OUTDIR, 'collection_summary.csv'), index=False)
    print(f"\nSummary -> {OUTDIR}/collection_summary.csv")

    print(f"\nNext step: run 01_merge_specz.py to build the unified database.")
