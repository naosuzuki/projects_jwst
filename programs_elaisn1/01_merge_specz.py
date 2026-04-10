#!/usr/bin/env python3
"""
Merge all collected spectroscopic redshift catalogs for ELAIS-N1
into a single unified database.

Designed to be fault-tolerant: if some catalogs failed to download
in step 00, this script will skip them and process whatever is available.

Steps:
  1. Scan data/ for CSV, TXT, and FITS files
  2. Try to extract RA, Dec, redshift from each file
  3. Concatenate all successful reads
  4. Cross-match to remove duplicates (1 arcsec matching radius)
  5. Output: unified CSV + FITS database

Usage:
  python 01_merge_specz.py
"""

import os
import sys
import glob
import traceback
import warnings
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

warnings.filterwarnings('ignore')

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MATCH_RADIUS_ARCSEC = 1.0

# ELAIS-N1 field parameters
RA_CENTER = 242.75
DEC_CENTER = 55.0
SEARCH_RADIUS = 2.0  # degrees

# Output column names
OUTCOLS = ['ra', 'dec', 'z_spec', 'z_err', 'obj_class', 'survey', 'reference']


# ═══════════════════════════════════════════════════════════════════════════
# Column-finding helpers
# ═══════════════════════════════════════════════════════════════════════════

def _find_col(columns, patterns, exact=None):
    """Find the first column matching any of the patterns (case-insensitive).
    Returns column name or None."""
    cols_lower = {c: c.lower() for c in columns}
    # Try exact matches first
    if exact:
        for c, cl in cols_lower.items():
            if cl in [e.lower() for e in exact]:
                return c
    # Then substring patterns
    for c, cl in cols_lower.items():
        for pat in patterns:
            if pat.lower() in cl:
                return c
    return None


def _find_ra(columns):
    return _find_col(columns,
                     ['raj2000', 'ra_j2000', 'target_ra', 'alpha_j2000', '_ra'],
                     exact=['ra', 'RA', 'RA(J2000)'])

def _find_dec(columns):
    return _find_col(columns,
                     ['dej2000', 'de_j2000', 'dec_j2000', 'target_dec',
                      'delta_j2000', '_dec', '_de'],
                     exact=['dec', 'DEC', 'Dec', 'DE', 'Dec(J2000)'])

def _find_z(columns):
    """Find spectroscopic redshift column, preferring spec-z over photo-z."""
    # Priority order: explicit spec-z names first
    for pattern_group in [
        ['z_spec', 'zspec', 'z_sp', 'zsp', 'specz', 'spec_z'],
        ['redshift'],
        ['z_best', 'zbest'],
    ]:
        col = _find_col(columns, pattern_group)
        if col:
            return col
    # Last resort: bare 'z' column
    return _find_col(columns, [], exact=['z', 'Z', 'z_spec'])

def _find_zerr(columns):
    return _find_col(columns,
                     ['zerr', 'z_err', 'e_z', 'ez', 'redshift_err',
                      'z_error', 'zerror', 'sig_z'])

def _find_class(columns):
    return _find_col(columns,
                     ['class', 'obj_class', 'objclass', 'spectype',
                      'classification', 'type', 'obj_type', 'subclass'])


# ═══════════════════════════════════════════════════════════════════════════
# Generic auto-detect reader (works for any tabular file)
# ═══════════════════════════════════════════════════════════════════════════

def _make_output(df, ra_col, dec_col, z_col, zerr_col, class_col,
                 survey, reference):
    """Build standardized output DataFrame from detected columns."""
    out = pd.DataFrame({
        'ra':  pd.to_numeric(df[ra_col], errors='coerce'),
        'dec': pd.to_numeric(df[dec_col], errors='coerce'),
        'z_spec': pd.to_numeric(df[z_col], errors='coerce'),
        'z_err': pd.to_numeric(df[zerr_col], errors='coerce') if zerr_col else np.nan,
        'obj_class': df[class_col].astype(str) if class_col else '',
        'survey': survey,
        'reference': reference,
    })
    # Clean up obj_class
    if 'obj_class' in out.columns:
        out['obj_class'] = out['obj_class'].replace({'nan': '', 'None': '', 'none': ''})
    # Drop rows with no valid coordinates or redshift
    out = out.dropna(subset=['ra', 'dec', 'z_spec'])
    # Keep only positive redshifts
    out = out[out['z_spec'] > 0].copy()
    return out


def auto_read_csv(filepath, survey='unknown', reference='unknown'):
    """Auto-detect columns and read any CSV/TSV/TXT file."""
    basename = os.path.basename(filepath)

    # Try different separators
    df = None
    for sep in [',', r'\s+', '\t', ';', '|']:
        try:
            df = pd.read_csv(filepath, sep=sep, comment='#', nrows=5)
            if len(df.columns) >= 3:
                df = pd.read_csv(filepath, sep=sep, comment='#')
                break
            df = None
        except Exception:
            df = None

    if df is None or len(df) < 1:
        print(f"  [SKIP] Could not parse {basename}")
        return pd.DataFrame(columns=OUTCOLS)

    print(f"  Columns ({len(df.columns)}): {list(df.columns)[:12]}{'...' if len(df.columns) > 12 else ''}")
    print(f"  Rows: {len(df)}")

    ra_col = _find_ra(df.columns)
    dec_col = _find_dec(df.columns)
    z_col = _find_z(df.columns)
    zerr_col = _find_zerr(df.columns)
    class_col = _find_class(df.columns)

    if not ra_col:
        print(f"  [SKIP] No RA column found in {basename}")
        return pd.DataFrame(columns=OUTCOLS)
    if not dec_col:
        print(f"  [SKIP] No Dec column found in {basename}")
        return pd.DataFrame(columns=OUTCOLS)
    if not z_col:
        print(f"  [SKIP] No redshift column found in {basename}")
        return pd.DataFrame(columns=OUTCOLS)

    print(f"  Using: RA={ra_col}, Dec={dec_col}, z={z_col}"
          f"{f', zerr={zerr_col}' if zerr_col else ''}"
          f"{f', class={class_col}' if class_col else ''}")

    return _make_output(df, ra_col, dec_col, z_col, zerr_col, class_col,
                        survey, reference)


def auto_read_fits(filepath, survey='unknown', reference='unknown'):
    """Auto-detect columns and read any FITS file."""
    basename = os.path.basename(filepath)

    # Try different HDUs
    df = None
    for hdu in [1, 2, 0]:
        try:
            t = Table.read(filepath, hdu=hdu)
            if len(t.columns) >= 3 and len(t) > 0:
                df = t.to_pandas()
                print(f"  Read HDU {hdu}: {len(df)} rows, {len(df.columns)} columns")
                break
        except Exception:
            continue

    if df is None:
        try:
            t = Table.read(filepath)
            df = t.to_pandas()
            print(f"  Read (auto): {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"  [SKIP] Cannot read {basename}: {e}")
            return pd.DataFrame(columns=OUTCOLS)

    print(f"  Columns: {list(df.columns)[:15]}{'...' if len(df.columns) > 15 else ''}")

    ra_col = _find_ra(df.columns)
    dec_col = _find_dec(df.columns)
    z_col = _find_z(df.columns)
    zerr_col = _find_zerr(df.columns)
    class_col = _find_class(df.columns)

    if not ra_col or not dec_col or not z_col:
        print(f"  [SKIP] Missing required columns in {basename}"
              f" (RA={ra_col}, Dec={dec_col}, z={z_col})")
        return pd.DataFrame(columns=OUTCOLS)

    print(f"  Using: RA={ra_col}, Dec={dec_col}, z={z_col}"
          f"{f', zerr={zerr_col}' if zerr_col else ''}"
          f"{f', class={class_col}' if class_col else ''}")

    return _make_output(df, ra_col, dec_col, z_col, zerr_col, class_col,
                        survey, reference)


# ═══════════════════════════════════════════════════════════════════════════
# Survey-specific readers (thin wrappers with known column mappings)
# ═══════════════════════════════════════════════════════════════════════════

def read_mmt_hectospec(filepath):
    """MMT/Hectospec (Cheng+2021) - whitespace-separated, quality flag."""
    df = pd.read_csv(filepath, sep=r'\s+', comment='#')
    # Quality filter: keep zq >= 3
    if 'zq' in df.columns:
        df = df[pd.to_numeric(df['zq'], errors='coerce') >= 3]
    return _make_output(df,
                        _find_ra(df.columns), _find_dec(df.columns),
                        _find_z(df.columns), _find_zerr(df.columns),
                        _find_class(df.columns),
                        'MMT/Hectospec', 'Cheng+2021')


def read_help_specz(filepath):
    """HELP dmu23 - quality flag filter."""
    df = pd.read_csv(filepath, comment='#')
    # Quality filter: keep Q >= 3
    qual_col = _find_col(df.columns, ['qual', 'z_qual', 'quality'])
    if qual_col:
        q = pd.to_numeric(df[qual_col], errors='coerce')
        df = df[q >= 3]
    return _make_output(df,
                        _find_ra(df.columns), _find_dec(df.columns),
                        _find_z(df.columns), _find_zerr(df.columns),
                        _find_class(df.columns),
                        'HELP', 'HELP dmu23')


# ═══════════════════════════════════════════════════════════════════════════
# Identify survey from filename
# ═══════════════════════════════════════════════════════════════════════════

def guess_survey(basename):
    """Guess survey name and reference from filename."""
    bl = basename.lower()
    if 'sdss' in bl:
        return 'SDSS', 'SDSS DR18'
    if 'desi' in bl:
        return 'DESI', 'DESI EDR/DR1'
    if 'ned' in bl:
        return 'NED', 'NED compilation'
    if 'hectospec' in bl or 'mmt' in bl:
        return 'MMT/Hectospec', 'Cheng+2021'
    if 'hedam' in bl or 'elais-n1-specz' in bl:
        return 'HELP', 'HELP dmu23'
    if 'hetdex' in bl or 'debski' in bl:
        return 'HETDEX', 'Debski+2024'
    if 'gonzalez' in bl or '405_2243' in bl:
        return 'SWIRE', 'Gonzalez-Solares+2011'
    if 'rowan' in bl or '428_1958' in bl:
        return 'SWIRE', 'Rowan-Robinson+2013'
    if 'duncan' in bl or '648_A4' in bl or '648_a4' in bl:
        return 'LoTSS-Deep', 'Duncan+2021'
    if 'kondapally' in bl or '648_A3' in bl or '648_a3' in bl:
        return 'LoTSS-Deep', 'Kondapally+2021'
    if 'arnaudova' in bl or '542_2245' in bl:
        return 'LoTSS-Deep', 'Arnaudova+2025'
    if 'shirley' in bl or '507_129' in bl:
        return 'HELP', 'Shirley+2021'
    if 'malek' in bl or '620_A50' in bl or '620_a50' in bl:
        return 'HELP', 'Malek+2018'
    if 'lofar' in bl or 'en1' in bl or 'lotss' in bl:
        return 'LoTSS-Deep', 'LoTSS Deep Fields'
    if 'vizier' in bl:
        parts = basename.replace('vizier_', '').replace('.csv', '')
        return parts, parts
    # Fallback
    name = os.path.splitext(basename)[0]
    return name, name


# ═══════════════════════════════════════════════════════════════════════════
# Deduplication
# ═══════════════════════════════════════════════════════════════════════════

def deduplicate(df, radius_arcsec=MATCH_RADIUS_ARCSEC):
    """Remove duplicates by positional cross-matching within radius_arcsec."""
    print(f"\nDeduplicating {len(df)} entries (match radius = {radius_arcsec} arcsec)...")

    if len(df) <= 1:
        return df

    survey_priority = {
        'DESI': 0, 'SDSS': 1, 'MMT/Hectospec': 2, 'HETDEX': 3,
        'SWIRE': 4, 'HELP': 5, 'LoTSS-Deep': 6, 'NED': 7
    }
    df = df.copy()
    df['_priority'] = df['survey'].map(survey_priority).fillna(8)
    df['_zerr_sort'] = df['z_err'].fillna(999)
    df = df.sort_values(['_zerr_sort', '_priority']).reset_index(drop=True)

    try:
        coords = SkyCoord(ra=df['ra'].values * u.deg, dec=df['dec'].values * u.deg)
        keep = np.ones(len(df), dtype=bool)

        idx1, idx2, sep, _ = coords.search_around_sky(
            coords, radius_arcsec * u.arcsec)
        mask = idx1 < idx2  # each pair once, keep lower index
        for i, j in zip(idx1[mask], idx2[mask]):
            if keep[i] and keep[j]:
                keep[j] = False

        df_dedup = df[keep].drop(columns=['_priority', '_zerr_sort']).reset_index(drop=True)
        n_removed = len(df) - len(df_dedup)
        print(f"  Removed {n_removed} duplicates, {len(df_dedup)} unique sources remain")
        return df_dedup
    except Exception as e:
        print(f"  Deduplication error: {e}")
        print(f"  Returning all {len(df)} entries without deduplication")
        return df.drop(columns=['_priority', '_zerr_sort']).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("MERGING SPECTROSCOPIC REDSHIFT CATALOGS FOR ELAIS-N1")
    print("=" * 60)
    print(f"Data directory: {DATADIR}")

    # Find all data files
    all_files = sorted(
        glob.glob(os.path.join(DATADIR, '*.csv')) +
        glob.glob(os.path.join(DATADIR, '*.txt')) +
        glob.glob(os.path.join(DATADIR, '*.fits'))
    )

    # Filter out output files and summary
    skip = ['collection_summary.csv', 'elaisn1_specz_database.csv',
            'elaisn1_specz_database.fits', 'elaisn1_specz.sql', '.gitkeep']
    all_files = [f for f in all_files
                 if os.path.basename(f) not in skip
                 and os.path.getsize(f) > 50]  # skip empty/tiny files

    print(f"Found {len(all_files)} data files to process:\n")
    for f in all_files:
        size = os.path.getsize(f) / 1024
        print(f"  {os.path.basename(f)} ({size:.0f} KB)")

    if not all_files:
        print("\nNo data files found! Run 00_collect_specz.py first.")
        sys.exit(1)

    # ── Process each file ─────────────────────────────────────────────────
    all_dfs = []
    succeeded = []
    failed = []

    for filepath in all_files:
        basename = os.path.basename(filepath)
        survey, reference = guess_survey(basename)
        ext = os.path.splitext(basename)[1].lower()

        print(f"\n{'─'*50}")
        print(f"Reading: {basename}  (survey={survey})")
        print(f"{'─'*50}")

        try:
            if ext == '.fits':
                df = auto_read_fits(filepath, survey, reference)
            elif 'Hectospec' in basename or 'mmt' in basename.lower():
                df = read_mmt_hectospec(filepath)
            elif 'hedam' in basename.lower() or 'ELAIS-N1-specz' in basename:
                df = read_help_specz(filepath)
            else:
                df = auto_read_csv(filepath, survey, reference)

            if df is not None and len(df) > 0:
                print(f"  >> SUCCESS: {len(df)} spec-z entries")
                all_dfs.append(df)
                succeeded.append((basename, len(df), survey))
            else:
                print(f"  >> EMPTY: no valid spec-z extracted")
                failed.append((basename, "no valid data"))
        except Exception as e:
            print(f"  >> FAILED: {e}")
            traceback.print_exc()
            failed.append((basename, str(e)))

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FILE PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"  Succeeded: {len(succeeded)}")
    for name, n, survey in succeeded:
        print(f"    {name}: {n} rows ({survey})")
    if failed:
        print(f"  Failed/Empty: {len(failed)}")
        for name, reason in failed:
            print(f"    {name}: {reason}")

    if not all_dfs:
        print("\nNo valid data extracted from any file!")
        print("Check the files in data/ and ensure they contain RA, Dec, z columns.")
        sys.exit(1)

    # ── Concatenate ───────────────────────────────────────────────────────
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal entries before deduplication: {len(combined)}")
    print(f"  By survey:")
    for survey, count in combined['survey'].value_counts().items():
        print(f"    {survey}: {count}")

    # ── Deduplicate ───────────────────────────────────────────────────────
    final = deduplicate(combined)

    # ── Filter to ELAIS-N1 field ──────────────────────────────────────────
    cos_dec = np.cos(np.radians(DEC_CENTER))
    mask = ((final['ra'] > RA_CENTER - SEARCH_RADIUS / cos_dec) &
            (final['ra'] < RA_CENTER + SEARCH_RADIUS / cos_dec) &
            (final['dec'] > DEC_CENTER - SEARCH_RADIUS) &
            (final['dec'] < DEC_CENTER + SEARCH_RADIUS))
    n_before = len(final)
    final = final[mask].reset_index(drop=True)
    print(f"  Field cut: {n_before} -> {len(final)} sources in ELAIS-N1")

    if len(final) == 0:
        print("\nWARNING: No sources within the ELAIS-N1 field after filtering!")
        print("Check that the input files contain data near RA=242.75, Dec=+55.0")
        sys.exit(1)

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

    # ── Save CSV ──────────────────────────────────────────────────────────
    outpath_csv = os.path.join(DATADIR, 'elaisn1_specz_database.csv')
    final.to_csv(outpath_csv, index=False)
    print(f"\nSaved: {outpath_csv}")

    # ── Save FITS ─────────────────────────────────────────────────────────
    outpath_fits = os.path.join(DATADIR, 'elaisn1_specz_database.fits')
    try:
        t = Table.from_pandas(final)
        t.write(outpath_fits, overwrite=True)
        print(f"Saved: {outpath_fits}")
    except Exception as e:
        print(f"FITS save error: {e} (CSV was saved successfully)")

    print(f"\nDone. Database has {len(final)} unique spectroscopic redshifts.")


if __name__ == '__main__':
    main()
