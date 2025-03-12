import pandas
from astropy.io import fits

datadir='/Users/suzuki/data/JWST/DJA/20250309/'

def read_1dspec(filename):
   fitsfile=datadir+filename
   hdul = fits.open(fitsfile)  
   data = hdul[1].data
   wave=data['wave']
   flux=data['flux']
   print(len(wave))
   #print(wave)
   for i in range(len(wave)):
      print(wave[i],flux[i])
   print(len(wave))


filename='abell2744-castellano1-v3_prism-clear_3073_14130.spec.fits'
read_1dspec(filename)


