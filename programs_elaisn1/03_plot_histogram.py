#!/usr/bin/env python3
"""
Plot histogram of spectroscopic redshifts in the ELAIS-N1 database.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CSV_FILE = os.path.join(DATADIR, 'elaisn1_specz_database.csv')

df = pd.read_csv(CSV_FILE)
print(f"Loaded {len(df)} spectroscopic redshifts")

fig, ax = plt.subplots(figsize=(10, 6))

# Histogram with bins of dz=0.05
bins = np.arange(0, df['z_spec'].max() + 0.1, 0.05)

# Color by survey if multiple
surveys = df['survey'].unique()
if len(surveys) > 1:
    colors = plt.cm.tab10(np.linspace(0, 1, len(surveys)))
    bottom = np.zeros(len(bins) - 1)
    for i, survey in enumerate(surveys):
        sub = df[df['survey'] == survey]['z_spec']
        counts, _ = np.histogram(sub, bins=bins)
        ax.bar(bins[:-1], counts, width=0.048, bottom=bottom,
               color=colors[i], edgecolor='black', linewidth=0.3,
               label=f'{survey} ({len(sub)})', alpha=0.85)
        bottom += counts
else:
    ax.hist(df['z_spec'], bins=bins, color='steelblue',
            edgecolor='black', linewidth=0.5, alpha=0.85,
            label=f'{surveys[0]} ({len(df)})')

ax.set_xlabel('Spectroscopic Redshift', fontsize=14)
ax.set_ylabel('Number of Sources', fontsize=14)
ax.set_title(f'ELAIS-N1 Spectroscopic Redshifts (N = {len(df)})', fontsize=16)
ax.legend(fontsize=11, loc='upper right')
ax.tick_params(labelsize=12)
ax.set_xlim(0, df['z_spec'].max() + 0.1)

# Add median line
median_z = df['z_spec'].median()
ax.axvline(median_z, color='red', linestyle='--', linewidth=1.5,
           label=f'median z = {median_z:.3f}')
ax.legend(fontsize=11, loc='upper right')

plt.tight_layout()

outpath = os.path.join(DATADIR, 'elaisn1_specz_histogram.png')
fig.savefig(outpath, dpi=150)
print(f"Saved: {outpath}")
plt.close()
