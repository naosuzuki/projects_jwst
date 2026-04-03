ELAIS-N1 Spectroscopic Redshift Database
=========================================

Field: ELAIS-N1 (European Large Area ISO Survey - North 1)
Center: RA = 16h 11m 00s = 242.75 deg, Dec = +55d 00m 00s (J2000)
Area: ~9 sq deg (core), search radius 2.0 deg

Pipeline:
  00_collect_specz.py  - Download/query all catalogs (requires internet)
  01_merge_specz.py    - Merge, deduplicate, build unified database

Requirements:
  pip install astroquery astropy pandas requests


CATALOG INVENTORY
=================

1. SDSS/BOSS DR18 (~600+ spec-z)
   - Includes BOSS ancillary program targeting LOFAR sources in ELAIS-N1
   - Access: SDSS CasJobs SQL query or astroquery.sdss
   - Reference: SDSS-III/IV collaboration

2. HETDEX-LOFAR (9,710 spec-z)
   - Blind IFU spectroscopy from Hobby-Eberly Telescope
   - 28,705 spectra extracted for LOFAR sources, 9,087 new redshifts
   - Zenodo: https://zenodo.org/records/14194635
   - Reference: Debski et al. 2024, ApJ 978, 101 (arXiv:2411.08974)

3. DESI EDR/DR1 (partial coverage)
   - Some ELAIS-N1 sources observed in DESI survey validation
   - Access: NOIRLab DataLab TAP (https://datalab.noirlab.edu/tap)
   - Tables: desi_edr.zpix, desi_dr1.zpix
   - Reference: DESI Collaboration 2023 (arXiv:2306.06308)

4. LoTSS Deep Fields DR1 (spec-z compilation within multi-wavelength catalog)
   - Photo-z + spec-z for optical sources in ELAIS-N1
   - CDS/VizieR: J/A+A/648/A4 (Duncan+2021)
   - Cross-match catalog: J/A+A/648/A3 (Kondapally+2021)
   - Website: https://lofar-surveys.org/deepfields_public_en1.html
   - References:
     Duncan et al. 2021, A&A 648, A4
     Kondapally et al. 2021, A&A 648, A3

5. Gonzalez-Solares et al. 2011 (289 spec-z)
   - Largest spectroscopic follow-up in SWIRE ELAIS-N1
   - GMOS and WIYN spectroscopy
   - VizieR: J/MNRAS/405/2243
   - Reference: Gonzalez-Solares et al. 2011, MNRAS 405, 2243

6. Rowan-Robinson et al. 2013 (SWIRE photo-z with some spec-z)
   - Revised SWIRE photometric redshifts, includes spec-z where available
   - VizieR: J/MNRAS/428/1958
   - Reference: Rowan-Robinson et al. 2013, MNRAS 428, 1958

7. HELP Project (dmu23 ELAIS-N1)
   - Merged spectroscopic catalog from 8 surveys:
     Berta+2007, SDSS DR13, Trichas+2010, Swinbank+2007,
     Rowan-Robinson+2013 (WIYN/Keck/Gemini), NED, UZC, Lacy+2013
   - HeDaM: https://hedam.lam.fr/HELP/dataproducts/dmu23/dmu23_ELAIS-N1/
   - Quality flags: Q=1 (no z) to Q=5 (reliable, high-quality spectrum)
   - Reference: Shirley et al. 2021, MNRAS 507, 129

8. NED (NASA/IPAC Extragalactic Database)
   - Heterogeneous compilation of all published redshifts
   - Cone search: RA=242.75, Dec=55.0, r=2.0 deg
   - Access: https://ned.ipac.caltech.edu/

9. Arnaudova et al. 2025
   - Probabilistic spectroscopic classifications for LOFAR Deep Fields
   - Includes independent DESI spectral fitting for ELAIS-N1 sources
   - Reference: Arnaudova et al. 2025, MNRAS 542, 2245

10. LoTSS Deep Fields DR2
    - 505 hours of LOFAR observations, 154,952 radio sources
    - Reference: Sabater et al. 2025, A&A 695, A80 (arXiv:2501.04093)

11. HETDEX Public Source Catalog
    - 220K+ sources including 50K+ Lyman-alpha emitters
    - Zenodo: https://zenodo.org/records/7448504
    - Reference: Mentuch Cooper et al. 2023

12. Additional/future surveys:
    - WEAVE-LOFAR: Will provide complete spectroscopy for all deep field
      radio sources (R=5000, 365-960nm). In progress.
    - DESI main survey: Ongoing, will extend coverage.


MANUAL DOWNLOAD INSTRUCTIONS
=============================

If 00_collect_specz.py cannot connect to remote services, download these
files manually and place them in the data/ subdirectory:

1. HETDEX-LOFAR FITS:
   https://zenodo.org/records/14194635 -> download all FITS files

2. HELP spec-z CSV:
   https://hedam.lam.fr/HELP/dataproducts/dmu23/dmu23_ELAIS-N1/
   -> download ELAIS-N1-specz-v2_hedam.csv

3. LoTSS Deep Fields:
   https://lofar-surveys.org/deepfields_public_en1.html
   -> download the optical/Spitzer merged catalog and cross-match catalog

4. SDSS CasJobs (https://skyserver.sdss.org/casjobs/):
   Run the SQL query in 00_collect_specz.py, export as CSV -> sdss_specz.csv

5. NED batch query:
   https://ned.ipac.caltech.edu/forms/nearposn.html
   -> RA=242.75, Dec=55.0, Search Radius=120 arcmin

Then run: python 01_merge_specz.py
