#!/usr/bin/env python3
"""
Download all accessible spectroscopic redshift data for ELAIS-N1.
Tries GitHub raw URLs and other accessible endpoints.
"""
import os
import urllib.request
import json
import time

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(OUTDIR, exist_ok=True)

def download(url, fname, desc):
    fpath = os.path.join(OUTDIR, fname)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 100:
        print(f"  [SKIP] {fname} already exists ({os.path.getsize(fpath)/1024:.0f} KB)")
        return fpath
    try:
        print(f"  Downloading {desc}...")
        urllib.request.urlretrieve(url, fpath)
        size = os.path.getsize(fpath)
        if size < 100:
            os.remove(fpath)
            print(f"  [FAIL] {fname}: too small ({size} bytes), removed")
            return None
        print(f"  [OK] {fname} ({size/1024:.0f} KB)")
        return fpath
    except Exception as e:
        print(f"  [FAIL] {fname}: {e}")
        if os.path.exists(fpath):
            os.remove(fpath)
        return None

print("=" * 60)
print("DOWNLOADING SPEC-Z DATA FOR ELAIS-N1")
print("=" * 60)

results = {}

# 1. MMT/Hectospec (already have this)
print("\n[1] MMT/Hectospec (Cheng+2021)")
r = download(
    "https://raw.githubusercontent.com/chengchengcode/low-redshift-galaxies/main/MMT.Hectospec.specz.v2.txt",
    "MMT_Hectospec_specz_v2.txt",
    "Cheng+2021 MMT/Hectospec catalog"
)
if r: results['MMT'] = r

# 2. Try SDSS via SkyServer CSV export (different URL patterns)
print("\n[2] SDSS SkyServer")
sdss_urls = [
    ("https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch?cmd=SELECT+specobjid,plate,mjd,fiberid,ra,dec,z+as+redshift,zerr+as+redshift_err,zwarning,class,subclass,survey,programname+FROM+SpecObj+WHERE+ra+BETWEEN+239.26+AND+246.24+AND+dec+BETWEEN+53.0+AND+57.0+AND+zwarning%3D0+AND+z+BETWEEN+0.001+AND+7&format=csv",
     "sdss_specz.csv", "SDSS DR18 SQL query (WS)"),
    ("https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch?cmd=SELECT+specobjid,plate,mjd,fiberid,ra,dec,z+as+redshift,zerr+as+redshift_err,zwarning,class,subclass,survey,programname+FROM+SpecObj+WHERE+ra+BETWEEN+239.26+AND+246.24+AND+dec+BETWEEN+53.0+AND+57.0+AND+zwarning%3D0+AND+z+BETWEEN+0.001+AND+7&format=csv",
     "sdss_specz.csv", "SDSS DR17 SQL query (WS)"),
]
for url, fname, desc in sdss_urls:
    r = download(url, fname, desc)
    if r:
        results['SDSS'] = r
        break

# 3. Try VizieR TAP via different mirrors
print("\n[3] VizieR TAP mirrors")
vizier_mirrors = [
    "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync",
    "https://vizier.cfa.harvard.edu/TAPVizieR/tap/sync",
    "https://tapvizier.nao.ac.jp/TAPVizieR/tap/sync",
]
# Gonzalez-Solares 2011 - 289 spec-z
for mirror in vizier_mirrors:
    q = "SELECT+*+FROM+%22J/MNRAS/405/2243/table3%22"
    url = f"{mirror}?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY={q}"
    r = download(url, "vizier_gonzalez_solares2011.csv",
                 f"Gonzalez-Solares+2011 via {mirror.split('//')[1].split('/')[0]}")
    if r:
        results['GS2011'] = r
        break

# Duncan+2021 EN1 photo-z catalog (has z_spec column)
for mirror in vizier_mirrors:
    q = "SELECT+TOP+100000+*+FROM+%22J/A+A/648/A4/en1%22+WHERE+1%3DCONTAINS(POINT(%27ICRS%27,RAJ2000,DEJ2000),CIRCLE(%27ICRS%27,242.75,55.0,2.0))"
    url = f"{mirror}?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY={q}"
    r = download(url, "vizier_duncan2021_en1.csv",
                 f"Duncan+2021 EN1 via {mirror.split('//')[1].split('/')[0]}")
    if r:
        results['Duncan2021'] = r
        break

# DESI DR1 on VizieR
for mirror in vizier_mirrors:
    q = "SELECT+TOP+50000+*+FROM+%22V/161/zpix%22+WHERE+1%3DCONTAINS(POINT(%27ICRS%27,target_ra,target_dec),CIRCLE(%27ICRS%27,242.75,55.0,2.0))+AND+zwarn%3D0+AND+z+BETWEEN+0.001+AND+7"
    url = f"{mirror}?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY={q}"
    r = download(url, "vizier_desi_dr1.csv",
                 f"DESI DR1 via {mirror.split('//')[1].split('/')[0]}")
    if r:
        results['DESI'] = r
        break

# Rowan-Robinson 2004 ELAIS band-merged
for mirror in vizier_mirrors:
    q = "SELECT+*+FROM+%22J/MNRAS/351/1290/catalog%22+WHERE+1%3DCONTAINS(POINT(%27ICRS%27,RAJ2000,DEJ2000),CIRCLE(%27ICRS%27,242.75,55.0,3.0))"
    url = f"{mirror}?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY={q}"
    r = download(url, "vizier_rowan_robinson2004.csv",
                 f"Rowan-Robinson+2004 via {mirror.split('//')[1].split('/')[0]}")
    if r:
        results['RR2004'] = r
        break

# SDSS on VizieR
for mirror in vizier_mirrors:
    q = "SELECT+TOP+50000+*+FROM+%22V/154/sdss16%22+WHERE+1%3DCONTAINS(POINT(%27ICRS%27,RA_ICRS,DE_ICRS),CIRCLE(%27ICRS%27,242.75,55.0,2.0))+AND+zsp+BETWEEN+0.001+AND+7"
    url = f"{mirror}?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY={q}"
    r = download(url, "vizier_sdss_dr16.csv",
                 f"SDSS DR16 via VizieR {mirror.split('//')[1].split('/')[0]}")
    if r:
        results['SDSS_VizieR'] = r
        break

# 4. Zenodo - HETDEX-LOFAR catalog
print("\n[4] Zenodo - HETDEX-LOFAR")
try:
    resp = urllib.request.urlopen("https://zenodo.org/api/records/14194635", timeout=30)
    record = json.loads(resp.read())
    for f in record.get('files', []):
        fname = f['key']
        dl_url = f['links']['self']
        r = download(dl_url, fname, f"HETDEX-LOFAR {fname}")
        if r:
            results[f'HETDEX_{fname}'] = r
except Exception as e:
    print(f"  [FAIL] Zenodo API: {e}")

# 5. LOFAR surveys website
print("\n[5] LOFAR Surveys website")
lofar_urls = [
    ("https://lofar-surveys.org/public/EN1_opt_spitzer_merged_vac_opt3as_irac4as_all_hpx_forpub.fits",
     "EN1_opt_spitzer_merged_vac.fits", "LoTSS DR1 optical/Spitzer VAC"),
    ("https://lofar-surveys.org/public/ELAIS-N1/en1_final_cross_match_catalogue-v1.0.fits",
     "en1_final_cross_match_catalogue-v1.0.fits", "LoTSS DR1 cross-match"),
]
for url, fname, desc in lofar_urls:
    r = download(url, fname, desc)
    if r:
        results[fname] = r

# 6. NED via different URL patterns
print("\n[6] NED")
ned_urls = [
    ("https://ned.ipac.caltech.edu/cgi-bin/objsearch?in_csys=Equatorial&in_equinox=J2000.0&lon=242.75d&lat=55.0d&radius=120.0&search_type=Near+Position+Search&out_csys=Equatorial&out_equinox=J2000.0&of=ascii_tab&img_stamp=NO",
     "ned_specz.txt", "NED cone search (ascii)"),
    ("https://ned.ipac.caltech.edu/cgi-bin/objsearch?in_csys=Equatorial&in_equinox=J2000.0&lon=242.75d&lat=55.0d&radius=120.0&search_type=Near+Position+Search&out_csys=Equatorial&out_equinox=J2000.0&of=xml_main&img_stamp=NO",
     "ned_specz.xml", "NED cone search (XML)"),
]
for url, fname, desc in ned_urls:
    r = download(url, fname, desc)
    if r:
        results['NED'] = r
        break

# 7. DESI via NOIRLab DataLab
print("\n[7] DESI via NOIRLab DataLab")
desi_url = "https://datalab.noirlab.edu/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY=SELECT+targetid,survey,program,target_ra+as+ra,target_dec+as+dec,z+as+redshift,zerr+as+redshift_err,spectype,zwarn+FROM+desi_edr.zpix+WHERE+target_ra+BETWEEN+239.26+AND+246.24+AND+target_dec+BETWEEN+53.0+AND+57.0+AND+zwarn%3D0+AND+z+BETWEEN+0.001+AND+7"
r = download(desi_url, "desi_edr_specz.csv", "DESI EDR via DataLab TAP")
if r: results['DESI_EDR'] = r

desi_dr1_url = "https://datalab.noirlab.edu/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY=SELECT+targetid,survey,program,target_ra+as+ra,target_dec+as+dec,z+as+redshift,zerr+as+redshift_err,spectype,zwarn+FROM+desi_dr1.zpix+WHERE+target_ra+BETWEEN+239.26+AND+246.24+AND+target_dec+BETWEEN+53.0+AND+57.0+AND+zwarn%3D0+AND+z+BETWEEN+0.001+AND+7"
r = download(desi_dr1_url, "desi_dr1_specz.csv", "DESI DR1 via DataLab TAP")
if r: results['DESI_DR1'] = r

# 8. HeDaM HELP
print("\n[8] HELP HeDaM")
hedam_urls = [
    ("https://hedam.lam.fr/HELP/dataproducts/dmu23/dmu23_ELAIS-N1/ELAIS-N1-specz-v2_hedam.csv",
     "ELAIS-N1-specz-v2_hedam.csv", "HELP dmu23 ELAIS-N1 (hedam)"),
    ("https://hedam.lam.fr/HELP/dataproducts/dmu23/dmu23_ELAIS-N1/ELAIS-N1-specz-v2.csv",
     "ELAIS-N1-specz-v2.csv", "HELP dmu23 ELAIS-N1"),
]
for url, fname, desc in hedam_urls:
    r = download(url, fname, desc)
    if r:
        results['HELP'] = r
        break

# Summary
print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)
for key, path in results.items():
    size = os.path.getsize(path)
    print(f"  {key}: {os.path.basename(path)} ({size/1024:.0f} KB)")
print(f"\nTotal: {len(results)} catalogs downloaded")
print(f"Files in {OUTDIR}:")
for f in sorted(os.listdir(OUTDIR)):
    fp = os.path.join(OUTDIR, f)
    if os.path.isfile(fp):
        print(f"  {f} ({os.path.getsize(fp)/1024:.0f} KB)")
