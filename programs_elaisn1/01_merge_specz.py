#!/usr/bin/env python3
"""
Merge all collected spectroscopic redshift catalogs for ELAIS-N1
into a single unified database.

Steps:
  1. Read each downloaded catalog (CSV or FITS)
  2. Extract RA, Dec, redshift, redshift_err, source survey, object class
  3. Concatenate all catalogs
  4. Cross-match to remove duplicates (1 arcsec matching radius)
  5. For duplicates, keep the measurement with smallest redshift error
  6. Output: unified CSV + FITS database

Usage:
  python 01_merge_specz.py
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

warnings.filterwarnings('ignore')

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MATCH_RADIUS_ARCSEC = 1.0  # cross-match radius for deduplication


def read_sdss(filepath):
    """Parse SDSS spectroscopic CSV."""
    df = pd.read_csv(filepath)
    out = pd.DataFrame({
        'ra': df['ra'],
        'dec': df['dec'],
        'z_spec': df['redshift'],
        'z_err': df.get('redshift_err', np.nan),
        'obj_class': df.get('class', ''),
        'survey': 'SDSS',
        'reference': 'SDSS DR18',
    })
    return out[out['z_spec'] > 0]


def read_ned(filepath):
    """Parse NED CSV."""
    df = pd.read_csv(filepath)
    # NED columns vary; look for common names
    ra_col = [c for c in df.columns if 'RA' in c.upper()][0]
    dec_col = [c for c in df.columns if 'DEC' in c.upper()][0]
    z_col = [c for c in df.columns if 'Redshift' in c or 'redshift' in c or 'z' == c.lower()][0]

    out = pd.DataFrame({
        'ra': pd.to_numeric(df[ra_col], errors='coerce'),
        'dec': pd.to_numeric(df[dec_col], errors='coerce'),
        'z_spec': pd.to_numeric(df[z_col], errors='coerce'),
        'z_err': np.nan,
        'obj_class': df.get('Type', ''),
        'survey': 'NED',
        'reference': 'NED compilation',
    })
    return out.dropna(subset=['ra', 'dec', 'z_spec'])


def read_desi(filepath):
    """Parse DESI CSV."""
    df = pd.read_csv(filepath)
    out = pd.DataFrame({
        'ra': df['ra'],
        'dec': df['dec'],
        'z_spec': df['redshift'],
        'z_err': df.get('redshift_err', np.nan),
        'obj_class': df.get('spectype', ''),
        'survey': 'DESI',
        'reference': 'DESI EDR/DR1',
    })
    return out[out['z_spec'] > 0]


def read_vizier_gonzalez_solares(filepath):
    """Parse Gonzalez-Solares+2011 SWIRE ELAIS-N1."""
    df = pd.read_csv(filepath)
    # Look for RA/Dec and redshift columns
    ra_col = [c for c in df.columns if 'RA' in c.upper() or '_RA' in c.upper()][0]
    dec_col = [c for c in df.columns if 'DE' in c.upper() or 'DEC' in c.upper()][0]
    z_cols = [c for c in df.columns if 'z' in c.lower() and 'flag' not in c.lower()]
    z_col = z_cols[0] if z_cols else None
    if z_col is None:
        print(f"  Warning: no redshift column found in {filepath}")
        return pd.DataFrame()

    out = pd.DataFrame({
        'ra': pd.to_numeric(df[ra_col], errors='coerce'),
        'dec': pd.to_numeric(df[dec_col], errors='coerce'),
        'z_spec': pd.to_numeric(df[z_col], errors='coerce'),
        'z_err': np.nan,
        'obj_class': '',
        'survey': 'SWIRE',
        'reference': 'Gonzalez-Solares+2011',
    })
    return out.dropna(subset=['ra', 'dec', 'z_spec']).query('z_spec > 0')


def read_vizier_generic(filepath, survey_name, reference):
    """Generic reader for VizieR CSV files - tries to find RA, Dec, z columns."""
    df = pd.read_csv(filepath)

    # Find RA column
    ra_candidates = [c for c in df.columns
                     if any(k in c.upper() for k in ['RAJ2000', 'RA_', '_RA', 'ALPHA'])]
    if not ra_candidates:
        ra_candidates = [c for c in df.columns if c.upper() in ('RA',)]
    if not ra_candidates:
        print(f"  Warning: no RA column in {filepath}: {list(df.columns)[:10]}")
        return pd.DataFrame()
    ra_col = ra_candidates[0]

    # Find Dec column
    dec_candidates = [c for c in df.columns
                      if any(k in c.upper() for k in ['DEJ2000', 'DEC_', '_DEC', 'DELTA', 'DE_'])]
    if not dec_candidates:
        dec_candidates = [c for c in df.columns if c.upper() in ('DEC', 'DE')]
    if not dec_candidates:
        print(f"  Warning: no Dec column in {filepath}")
        return pd.DataFrame()
    dec_col = dec_candidates[0]

    # Find spectroscopic redshift column
    z_candidates = [c for c in df.columns
                    if any(k in c.lower() for k in ['zspec', 'z_spec', 'zsp', 'z_sp',
                                                      'redshift', 'z_best'])]
    if not z_candidates:
        # Try just 'z' but be careful
        z_candidates = [c for c in df.columns if c.lower() in ('z',)]
    if not z_candidates:
        print(f"  Warning: no redshift column in {filepath}: {list(df.columns)[:15]}")
        return pd.DataFrame()
    z_col = z_candidates[0]

    # Find redshift error
    zerr_candidates = [c for c in df.columns
                       if any(k in c.lower() for k in ['zerr', 'z_err', 'e_z', 'ez'])]
    zerr_col = zerr_candidates[0] if zerr_candidates else None

    out = pd.DataFrame({
        'ra': pd.to_numeric(df[ra_col], errors='coerce'),
        'dec': pd.to_numeric(df[dec_col], errors='coerce'),
        'z_spec': pd.to_numeric(df[z_col], errors='coerce'),
        'z_err': pd.to_numeric(df[zerr_col], errors='coerce') if zerr_col else np.nan,
        'obj_class': '',
        'survey': survey_name,
        'reference': reference,
    })
    return out.dropna(subset=['ra', 'dec', 'z_spec']).query('z_spec > 0')


def read_help_specz(filepath):
    """Read HELP dmu23 ELAIS-N1 spec-z CSV.

    Columns: RA, DEC, Z_SPEC, Z_SOURCE, Z_QUAL, OBJID
    """
    df = pd.read_csv(filepath)
    print(f"  HELP columns: {list(df.columns)}")

    ra_col = [c for c in df.columns if 'ra' in c.lower()][0]
    dec_col = [c for c in df.columns if 'dec' in c.lower()][0]
    z_col = [c for c in df.columns if 'z_spec' in c.lower() or 'zspec' in c.lower()
             or c.lower() == 'z'][0]
    qual_col = [c for c in df.columns if 'qual' in c.lower()]

    out = pd.DataFrame({
        'ra': pd.to_numeric(df[ra_col], errors='coerce'),
        'dec': pd.to_numeric(df[dec_col], errors='coerce'),
        'z_spec': pd.to_numeric(df[z_col], errors='coerce'),
        'z_err': np.nan,
        'obj_class': '',
        'survey': 'HELP',
        'reference': 'HELP dmu23 (Berta+07/SDSS/Trichas+10/RR+13/Lacy+13)',
    })
    result = out.dropna(subset=['ra', 'dec', 'z_spec']).query('z_spec > 0')
    # Filter by quality if available (keep Q >= 3 = probable or better)
    if qual_col:
        q = pd.to_numeric(df[qual_col[0]], errors='coerce')
        good = q >= 3
        result = result[good.reindex(result.index, fill_value=True)]
    return result


def read_hetdex_lofar_fits(filepath):
    """Read HETDEX-LOFAR FITS catalog (Debski+2024).

    The FITS file has ext 1 = spectra, ext 2 = derived values including:
    RA, Dec, z_hetdex, classification, etc.
    """
    try:
        t = Table.read(filepath, hdu=2)
    except Exception:
        try:
            t = Table.read(filepath, hdu=1)
        except Exception:
            t = Table.read(filepath)

    df = t.to_pandas()
    print(f"  HETDEX-LOFAR columns: {list(df.columns)[:15]}")

    # Try to identify columns
    ra_col = [c for c in df.columns if 'ra' in c.lower()][0]
    dec_col = [c for c in df.columns if 'dec' in c.lower()][0]
    z_cols = [c for c in df.columns if 'z' in c.lower() and 'flag' not in c.lower()
              and 'err' not in c.lower() and 'warn' not in c.lower()]
    z_col = z_cols[0] if z_cols else None

    if z_col is None:
        print(f"  Warning: could not identify z column. Columns: {list(df.columns)}")
        return pd.DataFrame()

    out = pd.DataFrame({
        'ra': pd.to_numeric(df[ra_col], errors='coerce'),
        'dec': pd.to_numeric(df[dec_col], errors='coerce'),
        'z_spec': pd.to_numeric(df[z_col], errors='coerce'),
        'z_err': np.nan,
        'obj_class': df.get('classification', df.get('class', '')),
        'survey': 'HETDEX',
        'reference': 'Debski+2024',
    })
    return out.dropna(subset=['ra', 'dec', 'z_spec']).query('z_spec > 0')


def read_lofar_fits(filepath):
    """Read LoTSS Deep Fields FITS catalog and extract spec-z where available."""
    t = Table.read(filepath)
    df = t.to_pandas()
    print(f"  LOFAR catalog columns ({len(df.columns)} total): {list(df.columns)[:20]}...")

    # Look for spectroscopic redshift columns
    z_spec_cols = [c for c in df.columns
                   if any(k in c.lower() for k in ['z_spec', 'zspec', 'z_sp',
                                                     'specz', 'spec_z'])]
    if not z_spec_cols:
        # Check for z_best with a flag indicating spec-z
        z_best_cols = [c for c in df.columns if 'z_best' in c.lower()]
        flag_cols = [c for c in df.columns if 'z_source' in c.lower() or 'z_type' in c.lower()
                     or 'z_flag' in c.lower()]
        if z_best_cols and flag_cols:
            print(f"  Found z_best={z_best_cols[0]}, flag={flag_cols[0]}")
            z_spec_cols = z_best_cols  # will filter by flag below

    if not z_spec_cols:
        print(f"  No spec-z columns found in {os.path.basename(filepath)}")
        return pd.DataFrame()

    z_col = z_spec_cols[0]

    ra_col = [c for c in df.columns if c.upper() in ('RA', 'RAJ2000', 'RA_J2000')
              or 'ra' == c.lower()][0]
    dec_col = [c for c in df.columns if c.upper() in ('DEC', 'DEJ2000', 'DEC_J2000')
               or 'dec' == c.lower()][0]

    out = pd.DataFrame({
        'ra': pd.to_numeric(df[ra_col], errors='coerce'),
        'dec': pd.to_numeric(df[dec_col], errors='coerce'),
        'z_spec': pd.to_numeric(df[z_col], errors='coerce'),
        'z_err': np.nan,
        'obj_class': '',
        'survey': 'LoTSS-Deep',
        'reference': 'LoTSS Deep Fields DR1',
    })
    return out.dropna(subset=['ra', 'dec', 'z_spec']).query('z_spec > 0')


def deduplicate(df, radius_arcsec=MATCH_RADIUS_ARCSEC):
    """Remove duplicate entries by positional cross-matching.

    For groups of matches within radius_arcsec, keep the entry with the
    smallest redshift error. If errors are NaN, prefer entries from
    larger/more-reliable surveys.
    """
    print(f"\nDeduplicating {len(df)} entries (match radius = {radius_arcsec} arcsec)...")

    if len(df) == 0:
        return df

    # Sort by z_err (NaN last) so we keep best measurements
    survey_priority = {
        'DESI': 0, 'SDSS': 1, 'HETDEX': 2,
        'SWIRE': 3, 'HELP': 4, 'LoTSS-Deep': 5, 'NED': 6
    }
    df = df.copy()
    df['_priority'] = df['survey'].map(survey_priority).fillna(6)
    df['_zerr_sort'] = df['z_err'].fillna(999)
    df = df.sort_values(['_zerr_sort', '_priority']).reset_index(drop=True)

    coords = SkyCoord(ra=df['ra'].values * u.deg, dec=df['dec'].values * u.deg)

    # Use astropy matching
    keep = np.ones(len(df), dtype=bool)
    # Self-match to find groups
    idx, sep, _ = coords.match_to_catalog_sky(coords, nthneighbor=2)

    # Mark duplicates (keep first = best measurement due to sorting)
    matched = sep < radius_arcsec * u.arcsec
    for i in range(len(df)):
        if not keep[i]:
            continue
        if matched[i] and idx[i] < i:
            # This entry matches an earlier (better) entry -> remove
            keep[i] = False

    # More thorough: pairwise within small groups
    # Use search_around_sky for comprehensive dedup
    idx1, idx2, sep, _ = coords.search_around_sky(coords, radius_arcsec * u.arcsec)
    # Remove self-matches
    mask = idx1 != idx2
    idx1, idx2, sep = idx1[mask], idx2[mask], sep[mask]

    # For each pair, mark the lower-priority one
    for i, j in zip(idx1, idx2):
        if not keep[i] or not keep[j]:
            continue
        # Keep the one with lower index (better z_err / higher priority)
        keep[max(i, j)] = False

    df_dedup = df[keep].drop(columns=['_priority', '_zerr_sort']).reset_index(drop=True)
    n_removed = len(df) - len(df_dedup)
    print(f"  Removed {n_removed} duplicates, {len(df_dedup)} unique sources remain")
    return df_dedup


def main():
    print("=" * 60)
    print("MERGING SPECTROSCOPIC REDSHIFT CATALOGS FOR ELAIS-N1")
    print("=" * 60)

    all_dfs = []

    # ── CSV files from 00_collect_specz.py ────────────────────────────────
    csv_files = sorted(glob.glob(os.path.join(DATADIR, '*.csv')))
    for f in csv_files:
        basename = os.path.basename(f)
        if basename == 'collection_summary.csv' or basename.startswith('elaisn1_specz'):
            continue

        print(f"\nReading {basename}...")
        try:
            if 'sdss_specz' in basename:
                df = read_sdss(f)
            elif 'ned_specz' in basename:
                df = read_ned(f)
            elif 'desi_specz' in basename:
                df = read_desi(f)
            elif 'ELAIS-N1-specz' in basename or 'hedam' in basename:
                df = read_help_specz(f)
            elif 'vizier_J_MNRAS_405_2243' in basename:
                df = read_vizier_gonzalez_solares(f)
            elif 'vizier_' in basename:
                # Generic VizieR reader
                parts = basename.replace('vizier_', '').replace('.csv', '')
                df = read_vizier_generic(f, parts, parts)
            else:
                df = read_vizier_generic(f, basename, basename)

            if len(df) > 0:
                print(f"  -> {len(df)} spec-z entries")
                all_dfs.append(df)
        except Exception as e:
            print(f"  Error reading {basename}: {e}")

    # ── FITS files (HETDEX-LOFAR, LOFAR Deep Fields) ─────────────────────
    fits_files = sorted(glob.glob(os.path.join(DATADIR, '*.fits')))
    for f in fits_files:
        basename = os.path.basename(f)
        print(f"\nReading {basename}...")
        try:
            if 'hetdex' in basename.lower() or 'debski' in basename.lower():
                df = read_hetdex_lofar_fits(f)
            else:
                df = read_lofar_fits(f)

            if len(df) > 0:
                print(f"  -> {len(df)} spec-z entries")
                all_dfs.append(df)
        except Exception as e:
            print(f"  Error reading {basename}: {e}")

    if not all_dfs:
        print("\nNo data files found! Run 00_collect_specz.py first.")
        return

    # ── Concatenate ───────────────────────────────────────────────────────
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal entries before deduplication: {len(combined)}")
    print(f"  By survey:")
    for survey, count in combined['survey'].value_counts().items():
        print(f"    {survey}: {count}")

    # ── Deduplicate ───────────────────────────────────────────────────────
    final = deduplicate(combined)

    # ── Filter to ELAIS-N1 field ──────────────────────────────────────────
    ra_c, dec_c, rad = 242.75, 55.0, 2.0
    mask = ((final['ra'] > ra_c - rad / np.cos(np.radians(dec_c))) &
            (final['ra'] < ra_c + rad / np.cos(np.radians(dec_c))) &
            (final['dec'] > dec_c - rad) &
            (final['dec'] < dec_c + rad))
    final = final[mask].reset_index(drop=True)
    print(f"  After field cut: {len(final)} sources in ELAIS-N1")

    # ── Statistics ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL ELAIS-N1 SPECTROSCOPIC REDSHIFT DATABASE")
    print(f"{'='*60}")
    print(f"Total unique sources: {len(final)}")
    print(f"Redshift range: {final['z_spec'].min():.4f} - {final['z_spec'].max():.4f}")
    print(f"Median redshift: {final['z_spec'].median():.4f}")
    print(f"\nBy survey:")
    for survey, count in final['survey'].value_counts().items():
        print(f"  {survey}: {count}")
    print(f"\nBy reference:")
    for ref, count in final['reference'].value_counts().items():
        print(f"  {ref}: {count}")

    # ── Save ──────────────────────────────────────────────────────────────
    outpath_csv = os.path.join(DATADIR, 'elaisn1_specz_database.csv')
    final.to_csv(outpath_csv, index=False)
    print(f"\nSaved: {outpath_csv}")

    outpath_fits = os.path.join(DATADIR, 'elaisn1_specz_database.fits')
    t = Table.from_pandas(final)
    t.write(outpath_fits, overwrite=True)
    print(f"Saved: {outpath_fits}")

    print(f"\nDone. Database has {len(final)} unique spectroscopic redshifts.")


if __name__ == '__main__':
    main()
