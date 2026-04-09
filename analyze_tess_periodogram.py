#!/usr/bin/env python3
"""
TESS Photometric Analysis of J074502.2-245638 (Gaia DR3 5614593699906384768, G=14.8)

This script extracts photometry from TESS FFI cube files on the MAST S3 bucket
using targeted HTTP range requests (no MAST API access needed), then performs
Lomb-Scargle periodogram analysis and generates phase diagrams.

Data extraction method:
  - The TESS FFI cubes are stored as 4D FITS arrays on S3: s3://stpubdata/tess/public/mast/
  - For each sector, we use tess-point to find the pixel coordinates of the target
  - We download only the 7x7 pixel cutout around the target using HTTP Range headers
  - Each cutout requires only ~430 KB per sector (vs 40-400 GB for the full cube)

Requirements:
  pip install numpy scipy matplotlib astropy tess-point requests
"""

import numpy as np
import json
import requests
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import median_filter
from astropy.timeseries import LombScargle
from tess_stars2px import tess_stars2px_function_entry

# =============================================================================
# Target
# =============================================================================
TARGET_NAME = "J074502.2-245638"
GAIA_ID = "Gaia DR3 5614593699906384768"
G_MAG = 14.8
RA_DEG = 116.259167
DEC_DEG = -24.943889

# =============================================================================
# Step 1: Determine TESS sector coverage
# =============================================================================
print(f"Target: {TARGET_NAME} ({GAIA_ID}, G={G_MAG})")
print(f"Coordinates: RA={RA_DEG:.6f}, Dec={DEC_DEG:.6f}")
print("=" * 70)

result = tess_stars2px_function_entry(0, RA_DEG, DEC_DEG)
sectors = result[3]
cameras = result[4]
ccds = result[5]
colpix = result[6]
rowpix = result[7]

print(f"\nTESS sector coverage ({len(sectors)} sectors):")
for i in range(len(sectors)):
    print(f"  Sector {sectors[i]:3d}: Cam{cameras[i]}/CCD{ccds[i]}, "
          f"pixel ({colpix[i]:.1f}, {rowpix[i]:.1f})")

# =============================================================================
# Step 2: Extract photometry from FFI cubes via S3
# =============================================================================
print("\n" + "=" * 70)
print("EXTRACTING PHOTOMETRY FROM TESS FFI CUBES")
print("=" * 70)

session = requests.Session()
HALF_BOX = 3  # 7x7 pixel cutout

sector_data = {}

for idx in range(len(sectors)):
    sector = sectors[idx]
    cam = cameras[idx]
    ccd = ccds[idx]
    col = int(round(colpix[idx]))
    row = int(round(rowpix[idx]))

    cube_url = (f"https://stpubdata.s3.amazonaws.com/tess/public/mast/"
                f"tess-s{sector:04d}-{cam}-{ccd}-cube.fits")

    print(f"\n--- Sector {sector} (Cam{cam}/CCD{ccd}) pixel ({col},{row}) ---")

    try:
        # Check if cube exists
        resp_head = session.head(cube_url, timeout=10)
        if resp_head.status_code != 200:
            print(f"  Cube not available (HTTP {resp_head.status_code})")
            continue

        file_size = int(resp_head.headers.get('Content-Length', 0))
        print(f"  Cube size: {file_size/1e9:.2f} GB")

        # Read FITS header
        resp_hdr = session.get(cube_url, timeout=30,
                               headers={'Range': 'bytes=0-30000'})
        raw = resp_hdr.content

        # Parse extension header for dimensions and timing
        tstart = tstop = 0.0
        naxes = {}
        data_offset = 0
        in_ext = False

        for block_start in range(0, len(raw), 2880):
            block = raw[block_start:block_start+2880]
            found_end = False
            for card_start in range(0, 2880, 80):
                card = block[card_start:card_start+80].decode('ascii', errors='replace')
                if card.startswith('XTENSION'):
                    in_ext = True
                if in_ext:
                    m = re.match(r'NAXIS(\d)\s*=\s*(\d+)', card)
                    if m:
                        naxes[int(m.group(1))] = int(m.group(2))
                    m = re.match(r'TSTART\s*=\s*([\d.Ee+-]+)', card)
                    if m:
                        tstart = float(m.group(1))
                    m = re.match(r'TSTOP\s*=\s*([\d.Ee+-]+)', card)
                    if m:
                        tstop = float(m.group(1))
                if card.startswith('END') and in_ext:
                    data_offset = block_start + 2880
                    found_end = True
                    break
            if found_end:
                break

        # Also check primary header for TSTART/TSTOP
        for i in range(0, min(2880, len(raw)), 80):
            card = raw[i:i+80].decode('ascii', errors='replace')
            m = re.match(r'TSTART\s*=\s*([\d.Ee+-]+)', card)
            if m and tstart == 0:
                tstart = float(m.group(1))
            m = re.match(r'TSTOP\s*=\s*([\d.Ee+-]+)', card)
            if m and tstop == 0:
                tstop = float(m.group(1))

        n1 = naxes.get(1, 2)
        n2 = naxes.get(2, 0)
        n3 = naxes.get(3, 0)
        n4 = naxes.get(4, 0)

        if n2 == 0 or n3 == 0 or n4 == 0:
            print("  Failed to parse cube dimensions")
            continue

        cadence_sec = (tstop - tstart) * 86400 / n2
        print(f"  Dimensions: {n4}x{n3}x{n2}x{n1}, cadence={cadence_sec:.0f}s")
        print(f"  Time: {tstart:.3f} to {tstop:.3f} BTJD")

        # Download 7x7 cutout via range requests
        col_min = max(0, col - HALF_BOX)
        col_max = min(n4 - 1, col + HALF_BOX)
        row_min = max(0, row - HALF_BOX)
        row_max = min(n3 - 1, row + HALF_BOX)
        actual_cols = col_max - col_min + 1
        actual_rows = row_max - row_min + 1

        cutout = np.zeros((actual_cols, actual_rows, n2, n1), dtype=np.float32)
        download_ok = True

        for ci, c in enumerate(range(col_min, col_max + 1)):
            flat_start = n1 * (0 + n2 * (row_min + n3 * c))
            byte_start = data_offset + flat_start * 4
            n_bytes = actual_rows * n2 * n1 * 4
            byte_end = byte_start + n_bytes - 1

            resp_data = session.get(cube_url, timeout=120,
                                    headers={'Range': f'bytes={byte_start}-{byte_end}'})
            if resp_data.status_code == 206 and len(resp_data.content) == n_bytes:
                arr = np.frombuffer(resp_data.content, dtype='>f4')
                cutout[ci] = arr.reshape(actual_rows, n2, n1)
            else:
                print(f"    Download error at col {c}")
                download_ok = False

        if not download_ok:
            continue

        # Photometry
        times = np.linspace(tstart, tstop, n2)
        center_ci = col - col_min
        center_ri = row - row_min

        # Target pixel flux (plane 0 = flux)
        target_flux = cutout[center_ci, center_ri, :, 0]

        # 3x3 aperture
        r1, r2 = max(0, center_ri-1), min(actual_rows, center_ri+2)
        c1, c2 = max(0, center_ci-1), min(actual_cols, center_ci+2)
        aper_flux = cutout[c1:c2, r1:r2, :, 0].sum(axis=(0, 1))

        # Background from outer ring
        bg_mask = np.ones((actual_cols, actual_rows), dtype=bool)
        bg_mask[c1:c2, r1:r2] = False
        bg_flux = cutout[:, :, :, 0][bg_mask].mean(axis=0)

        # Background-subtracted aperture
        n_aper_pix = (c2-c1) * (r2-r1)
        flux_sub = aper_flux - n_aper_pix * bg_flux

        # Quality filtering
        bg_median = np.median(bg_flux)
        bg_mad = np.median(np.abs(bg_flux - bg_median)) * 1.4826
        bg_good = np.abs(bg_flux - bg_median) < 3 * bg_mad

        fs_median = np.median(flux_sub[bg_good])
        fs_mad = np.median(np.abs(flux_sub[bg_good] - fs_median)) * 1.4826
        fs_good = np.abs(flux_sub - fs_median) < 5 * fs_mad
        good = bg_good & fs_good

        t_good = times[good]
        f_good = flux_sub[good]

        # Detrend with moving median
        window = max(int(len(f_good) * 0.05), 51)
        if window % 2 == 0:
            window += 1
        trend = median_filter(f_good, size=window)
        f_detrended = f_good / trend

        # Sigma clip detrended
        dt_med = np.nanmedian(f_detrended)
        dt_mad = np.nanmedian(np.abs(f_detrended - dt_med)) * 1.4826
        clip = np.abs(f_detrended - dt_med) < 4 * dt_mad

        sector_data[sector] = {
            'times': t_good[clip],
            'flux': f_detrended[clip],
            'n_frames_total': n2,
            'n_frames_good': int(clip.sum()),
            'cadence_sec': cadence_sec,
            'tstart': tstart,
            'tstop': tstop,
        }

        rms = np.std(f_detrended[clip]) * 100
        print(f"  Good frames: {clip.sum()}/{n2} ({100*clip.sum()/n2:.1f}%)")
        print(f"  Detrended RMS: {rms:.3f}%")
        print(f"  SUCCESS")

    except Exception as e:
        print(f"  Error: {e}")

print(f"\n{'='*70}")
print(f"Successfully extracted {len(sector_data)} sectors: {sorted(sector_data.keys())}")

# =============================================================================
# Step 3: Periodogram Analysis
# =============================================================================
print(f"\n{'='*70}")
print("LOMB-SCARGLE PERIODOGRAM ANALYSIS")
print(f"{'='*70}")

combined_t = np.concatenate([sector_data[s]['times'] for s in sorted(sector_data.keys())])
combined_f = np.concatenate([sector_data[s]['flux'] for s in sorted(sector_data.keys())])
print(f"Combined: {len(combined_t)} data points, baseline={combined_t.max()-combined_t.min():.1f} days")

# Use best-cadence sector for short-period search
best_sector = max(sector_data.keys(), key=lambda s: sector_data[s]['n_frames_good'])
t_best = sector_data[best_sector]['times']
f_best = sector_data[best_sector]['flux'] - 1.0

# Short-period periodogram
freq_short = np.linspace(0.5, 400, 200000)
ls_best = LombScargle(t_best, f_best)
power_short = ls_best.power(freq_short)
periods_short = 1.0 / freq_short
fap_short = ls_best.false_alarm_level([0.1, 0.01, 0.001])

print(f"\n--- Short Period Search (Sector {best_sector}, {sector_data[best_sector]['cadence_sec']:.0f}s cadence) ---")
print(f"FAP levels: 10%={fap_short[0]:.4f}, 1%={fap_short[1]:.4f}, 0.1%={fap_short[2]:.4f}")

peak_idx = signal.find_peaks(power_short, height=fap_short[2], distance=100)[0]
candidate_periods = []
if len(peak_idx) > 0:
    top = peak_idx[np.argsort(power_short[peak_idx])[-10:]]
    print("Significant peaks (>0.1% FAP):")
    for p in sorted(top, key=lambda x: power_short[x], reverse=True):
        per = periods_short[p]
        pwr = power_short[p]
        fap = ls_best.false_alarm_probability(pwr)
        print(f"  P = {per:.6f} d ({per*24:.4f} hr), power={pwr:.4f}, FAP={fap:.1e}")
        candidate_periods.append(per)

# Long-period periodogram (combined)
freq_long = np.linspace(0.05, 50, 200000)
ls_comb = LombScargle(combined_t, combined_f - 1.0)
power_long = ls_comb.power(freq_long)
periods_long = 1.0 / freq_long

print(f"\n--- Long Period Search (Combined) ---")
peak_idx_l = signal.find_peaks(power_long, height=0.001, distance=200)[0]
if len(peak_idx_l) > 0:
    top_l = peak_idx_l[np.argsort(power_long[peak_idx_l])[-5:]]
    for p in sorted(top_l, key=lambda x: power_long[x], reverse=True):
        per = periods_long[p]
        pwr = power_long[p]
        fap = ls_comb.false_alarm_probability(pwr)
        print(f"  P = {per:.6f} d ({per*24:.4f} hr), power={pwr:.4f}, FAP={fap:.1e}")

# =============================================================================
# Step 4: Plots
# =============================================================================
# Summary figure
fig = plt.figure(figsize=(18, 22))
gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.25)
fig.suptitle(f'TESS Time-Series Analysis of {TARGET_NAME}\n({GAIA_ID}, G={G_MAG})',
             fontsize=15, y=0.98)

# Light curves
ax_lc = fig.add_subplot(gs[0, :])
colors_map = {7: 'C0', 34: 'C1', 61: 'C2', 88: 'C3'}
for s in sorted(sector_data.keys()):
    c = colors_map.get(s, f'C{s % 10}')
    ax_lc.plot(sector_data[s]['times'], sector_data[s]['flux'],
               '.', ms=0.2, alpha=0.3, color=c, label=f'Sector {s}')
ax_lc.set_xlabel('Time (BTJD)')
ax_lc.set_ylabel('Relative Flux')
ax_lc.set_title('Detrended Light Curves')
ax_lc.legend(markerscale=20, fontsize=9)
ax_lc.set_ylim(0.97, 1.03)

# Periodogram (short)
ax_pg1 = fig.add_subplot(gs[1, 0])
ax_pg1.plot(periods_short * 24, power_short, '-', lw=0.3, color='C3')
ax_pg1.axhline(fap_short[1], color='orange', ls='--', lw=0.8, label='1% FAP')
ax_pg1.axhline(fap_short[2], color='red', ls='--', lw=0.8, label='0.1% FAP')
ax_pg1.set_xlabel('Period (hours)')
ax_pg1.set_ylabel('L-S Power')
ax_pg1.set_title(f'Periodogram: Sector {best_sector}')
ax_pg1.set_xlim(0, 48)
ax_pg1.legend(fontsize=8)

# Periodogram (long)
ax_pg2 = fig.add_subplot(gs[1, 1])
fap_comb = ls_comb.false_alarm_level([0.01, 0.001])
ax_pg2.plot(periods_long, power_long, '-', lw=0.3, color='black')
ax_pg2.axhline(fap_comb[0], color='orange', ls='--', lw=0.8, label='1% FAP')
ax_pg2.axhline(fap_comb[1], color='red', ls='--', lw=0.8, label='0.1% FAP')
ax_pg2.set_xlabel('Period (days)')
ax_pg2.set_ylabel('L-S Power')
ax_pg2.set_title('Periodogram: Combined')
ax_pg2.set_xlim(0, 5)
ax_pg2.legend(fontsize=8)

# Phase diagrams for top 3 candidates
top3 = candidate_periods[:3] if len(candidate_periods) >= 3 else candidate_periods
for row_idx, period in enumerate(top3):
    for col_idx, (use_sectors, title_suffix) in enumerate([
        (sorted(sector_data.keys()), 'All sectors'),
        ([best_sector], f'Sector {best_sector}')
    ]):
        ax = fig.add_subplot(gs[row_idx + 2, col_idx])
        all_t = np.concatenate([sector_data[s]['times'] for s in use_sectors])
        all_f = np.concatenate([sector_data[s]['flux'] for s in use_sectors])
        phase = ((all_t - all_t.min()) / period) % 1.0

        ax.plot(phase, all_f, '.', ms=0.3, alpha=0.2, color='gray')
        ax.plot(phase + 1, all_f, '.', ms=0.3, alpha=0.2, color='gray')

        # Binned curve
        n_bins = 40
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bm = np.full(n_bins, np.nan)
        be = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = (phase >= bin_edges[b]) & (phase < bin_edges[b+1])
            if mask.sum() > 5:
                bm[b] = np.nanmedian(all_f[mask])
                be[b] = np.nanstd(all_f[mask]) / np.sqrt(mask.sum())

        ax.errorbar(bin_centers, bm, yerr=be, fmt='o-', color='black',
                   ms=4, lw=1.5, capsize=2, zorder=10)
        ax.errorbar(bin_centers + 1, bm, yerr=be, fmt='o-', color='black',
                   ms=4, lw=1.5, capsize=2, zorder=10)
        ax.set_xlim(-0.05, 2.05)
        ax.set_xlabel('Phase')
        ax.set_ylabel('Relative Flux')
        ax.set_title(f'P = {period:.4f} d ({period*24:.2f} hr) [{title_suffix}]', fontsize=10)
        yl = np.nanmax(np.abs(bm[np.isfinite(bm)] - 1)) * 5
        ax.set_ylim(1 - max(yl, 0.005), 1 + max(yl, 0.005))

plt.savefig('tess_analysis_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: tess_analysis_summary.png")
print("Done.")
