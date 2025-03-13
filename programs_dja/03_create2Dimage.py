import pandas as pd
from astropy.io import fits
import sys
import numpy
import astrolib

datadir='/Users/suzuki/data/JWST/DJA/20250309/'

def readtbl(csvfile):
   df=pd.read_csv(csvfile)
   print(df)
   df1=df.sort_values(by=['zfit'],ascending=True)
   prismimgall=numpy.zeros((7136,435))

   filenamelist=[] ; zlist=[]

   for i in range(len(df1)):
      filename=df1['file'].iloc[i]
      z=df1['zfit'].iloc[i]
      [npix,wave,flux,err]=read_1dspec(filename)

      negativeflag=numpy.where(flux<0.0,1,0)
      naflag=numpy.where(numpy.isnan(flux),1,0)

      negativepix=numpy.sum(negativeflag)
      napix=numpy.sum(naflag)

      print(i,'negative=',negativepix,'NaN',napix)
      #if(napix<235): filenamelist.append(filename)
      #if(napix<435): filenamelist.append(filename)
      filenamelist.append(filename)
      zlist.append(z)

   prismimg=numpy.zeros((len(filenamelist),435))
   prismimgrest=numpy.zeros((len(filenamelist),891))

   for i in range(len(filenamelist)):
      filename=filenamelist[i]
      [npix,wave,flux,err]=read_1dspec(filename)
      wavex=wave*10000.0
      wavex/=1.0+zlist[i]

# Check the wavelength interval
      logwave=numpy.log10(wavex)
      dlog=numpy.zeros(len(wavex))
      for j in range(len(wavex)-1):
        dlog[j]=logwave[j+1]-logwave[j]
      dlogave=numpy.average(dlog)
      COEFF0_obs=logwave[0]
      COEFF1_obs=dlogave

# Flux Normalization
      flux1=numpy.where(numpy.isnan(flux),0.0,flux)
      pixflag=numpy.where(flux>0.0,1,0)
      flux2=numpy.compress(pixflag,flux1)
      fluxmed=numpy.median(flux2)
      flux1/=fluxmed

# Restframe
      COEFF0_rest=0.0022157
      COEFF1_rest=1220.*COEFF0_rest
      [restwave,restflux,rid]=astrolib.exec_logrebinning_flux_fromlinear(COEFF0_rest,wavex,flux1)

      for j in range(435):
         prismimg[i,j]=flux1[j]
      print(i,"z=",df1['z'].iloc[i],"zfit=",df1['zfit'].iloc[i],filename,npix,negativepix,napix,dlogave,rid[0],rid[-1])

      for j in range(len(rid)):
         prismimgrest[i,rid[j]-1220]=restflux[j]
   
   outputfits='djaprism.fits'
   hdu=fits.PrimaryHDU(data=prismimg)
   hdul=fits.HDUList([hdu])
   hdr=hdul[0].header
   hdu.header['HISTORY']='Writing 2D FITS IMAGE : JWST PRISM'
   hdu.header['']='Wavelength (Ang) bin in Log, w=10**(coeff0+i*coeff1)'
   hdu.header['']='by Nao Suzuki on Mar 12th 2025'
   hdu.header['COEFF0']=COEFF0_obs
   hdu.header['COEFF1']=COEFF1_obs
   hdu.writeto(outputfits)
   del hdu ; del hdul
#  Original Version
#   writeoutimg(prismimg,outputfits)

   outputfits='djaprism_rest.fits'
   hdu=fits.PrimaryHDU(data=prismimgrest)
   hdul=fits.HDUList([hdu])
   hdr=hdul[0].header
   hdu.header['HISTORY']='Writing 2D FITS IMAGE : JWST PRISM'
   hdu.header['']='Wavelength (Ang) bin in Log, w=10**(coeff0+i*coeff1)'
   hdu.header['']='by Nao Suzuki on Mar 12th 2025'
   hdu.header['COEFF0']=COEFF0_rest
   hdu.header['COEFF1']=COEFF1_rest
   hdu.writeto(outputfits)
   del hdu ; del hdul
   sys.exit(1)
# Original Version w/o header
#   writeoutimg(prismimgrest,outputfits)


def writeoutimg(image2d,outputfits):
   hdu = fits.PrimaryHDU(data=image2d)
   hdul = fits.HDUList([hdu])
   hdul.writeto(outputfits)

def read_1dspec(filename):
# Reading FITS Table
   fitsfile=datadir+filename
   hdul = fits.open(fitsfile)
   hdr=hdul[1].header
   npix=hdr['NAXIS2']
   data = hdul[1].data
   wave=data['wave']
   flux=data['flux']
   err=data['err']
   return [npix,wave,flux,err]

csvfile='../csvfiles/dja_20250309.csv'
csvfile='../csvfiles/dja_20250309_prism.csv'
readtbl(csvfile)
