import sys
#import commands
import numpy
import math

"""
[mwebv]=find_MWEBV(radeg,decdeg) : 
"""

class Reddening:
    """
    Reddening class returns Schlegel-Cardelli
    reddening corrections for a given E(B-V)
    and optionally R_v.

    :Authors:
        - ``M. J. Childress <mjchildress@lbl.gov>``
        - ``R. C. Thomas    <rcthomas@lbl.gov>``
        - ``S. Bongard      <sbongard@lbl.gov>``
    :organization: ``The Nearby Supernova Factory`` http://snfactory.lbl.gov/
    :copyright: 2007
    """

    def __init__( self, EBV = 0.0, Rv   = 3.1 ) :
        """
        Evaluates a Schlegel-Cardelli type reddening correction,
        as defined by the parameters E(B-V) 'EBV', and R_v 'Rv'.

        :Parameters:
            `EBV` : float
                This sets E(B-V) value.
            `Rv` : float
                This sets R_v value.

        :return: ProSpect.Warp.Reddening object
        """
        self.EBV=EBV
        self.Rv=Rv

#-------------------------
#- Reddening Law Methods:
#-------------------------

#- CCM law reddening warp
    def ccm( self, wavelength ):
        """
        Given a wavelength in angstroms, calculate the
        extinction A(w) according to the CCM model.
        c.f. Cardelli, Clayton, Mathis 1989: ApJ 345:245

        :Parameters:
            `wavelength` : numarray.array
                Array of wavelength values at which you which to
                find warp values

        :return: numarray.array
        """
        # set the parameters
        Rv   = self.Rv
        EBV = self.EBV

        # change variables from angstroms to microns
        ang_to_micron = 1.0e-4
        # cardelli's parameter x = 1/(wavelength in microns)
        x = 1.0 / ( wavelength * ang_to_micron )

        # version to deal with arrays
        ex1 = ( x   < 1.1  ) * self.ccm_ir( x )
        ex2 = ( 1.1 <= x   ) * ( x < 3.3 ) * self.ccm_optical( x )
        ex3 = ( 3.3 <= x   ) * ( x < 8.0 ) * self.ccm_uv1( x )
        ex4 = ( x   >= 8.0 ) * self.ccm_uv2( x )
        extinction =  ex1 + ex2 + ex3 + ex4

        # get flux multiplicative factor from the extinction
        return 10.0**( -1.0* extinction * EBV / 2.5 )

    def ccm_ir( self, x):
        Rv=self.Rv
        # for x < 1.1
        a0 = 0.574
        b0 = -0.527
        c0 = Rv * a0 + b0
        return c0 * ( x**1.61 )

    def ccm_optical( self,x):
        Rv=self.Rv
        # for 1.1 < x < 3.3
        y = x - 1.82
        # a coeffs
        a0 =  1.0
        a1 =  0.17699
        a2 = -0.50447
        a3 = -0.02427
        a4 =  0.72085
        a5 =  0.01979
        a6 = -0.77530
        a7 =  0.32999
        # b coeffs
        b0 =  0.0
        b1 =  1.41338
        b2 =  2.28305
        b3 =  1.07233
        b4 = -5.38434
        b5 = -0.62251
        b6 =  5.30260
        b7 = -2.09002
        # total coeffs 
        c0 = Rv * a0 + b0
        c1 = Rv * a1 + b1
        c2 = Rv * a2 + b2
        c3 = Rv * a3 + b3
        c4 = Rv * a4 + b4
        c5 = Rv * a5 + b5
        c6 = Rv * a6 + b6
        c7 = Rv * a7 + b7
        # finally result
        return ( c0
                 + c1 * y 
                 + c2 * ( y**2 )
                 + c3 * ( y**3 )
                 + c4 * ( y**4 )
                 + c5 * ( y**5 )
                 + c6 * ( y**6 )
                 + c7 * ( y**7 ) )

    def ccm_uv1( self, x ):
        # for 3.3 < x < 8.0
        Rv=self.Rv
        y  = x - 5.90
        fa = ( x >= 5.90 )*( -0.04473 * ( y**2 ) - 0.009779 * ( y**3 ) )
        fb = ( x >= 5.90 )*(  0.21300 * ( y**2 ) + 0.120700 * ( y**3 ) )
        aa = 1.752 - 0.316 * x - 0.104 / ( ( x - 4.67 )**2 + 0.341 ) + fa
        bb = -3.09 + 1.825 * x + 1.206 / ( ( x - 4.62 )**2 + 0.263 ) + fb
        return Rv * aa + bb

    def ccm_uv2( self, x ):
        Rv=self.Rv
        # for x>8.0
        y = x - 8.0
        # a coeffs
        a0 = -1.073
        a1 = -0.628
        a2 =  0.137
        a3 = -0.070
        # b coeffs
        b0 = 13.670
        b1 =  4.257
        b2 = -0.420
        b3 =  0.374
        # total coeffs 
        c0 = Rv * a0 + b0
        c1 = Rv * a1 + b1
        c2 = Rv * a2 + b2
        c3 = Rv * a3 + b3
        return ( c0
                 + c1 * y
                 + c2 * ( y**2 )
                 + c3 * ( y**3 ) )

#def find_MWEBV(radeg,decdeg):
# Nao Suzuki
# Nov, 27 (Thu) 2008 Thanks Giving Day yeah!
# find E(B-V) from Schlegel table from RA & DEC in degrees
#
#    radegstr=str("%12.5f"%(radeg))+'d'
#    decdegstr=str("%12.5f"%(decdeg))+'d'
#    ned_query="\'http://nedwww.ipac.caltech.edu/cgi-bin/nph-calc?in_csys=\
#               Equatorial&in_equinox=J2000.0&obs_epoch=2000.0&\
#               lon="+radegstr+"&lat="+decdegstr+"&pa=0.0&\
#               out_csys=Equatorial&out_equinox=J2000.0\'"
#    nedlist=commands.getoutput('GET '+ned_query)
#    stringposition=nedlist.find("E(B-V)")
#    mwebv=eval(nedlist[stringposition+9:stringposition+15])
#    return [mwebv]

#def find_MWEBVfromNED(radeg,decdeg):
# Nao Suzuki
# Jun, 11 (Mon) 2011 : New Extinction DB from NED
# 2011 Schafly et al. vs 1998 Schlegel et al.
# Nov, 27 (Thu) 2008 Thanks Giving Day yeah!
# find E(B-V) from Schlegel table from RA & DEC in degrees
#
#    radegstr=str("%12.5f"%(radeg))+'d'
#    decdegstr=str("%12.5f"%(decdeg))+'d'
#    ned_query="\'http://nedwww.ipac.caltech.edu/cgi-bin/nph-calc?in_csys=\
#               Equatorial&in_equinox=J2000.0&obs_epoch=2000.0&\
#               lon="+radegstr+"&lat="+decdegstr+"&pa=0.0&\
#               out_csys=Equatorial&out_equinox=J2000.0\'"
#    nedlist=commands.getoutput('GET '+ned_query)
#    filterlist=["Landolt U (0.35)",\
#                "Landolt B (0.43)",\
#                "Landolt V (0.54)",\
#                "Landolt R (0.64)",\
#                "Landolt I (0.80)",\
#                "SDSS    u (0.36)",\
#                "SDSS    g (0.47)",\
#                "SDSS    r (0.62)",\
#                "SDSS    i (0.75)",\
#                "SDSS    z (0.89)",\
#                "UKIRT   J (1.25)",\
#                "UKIRT   H (1.66)",\
#                "UKIRT   K (2.19)",\
#                "UKIRT   L'(3.78)"]

#    landolt11=numpy.zeros(5) # Landolt UBVRI for Schafly  2011
#    landolt98=numpy.zeros(5) # Landolt UBVRI for Schlegel 1998
#    sdss11=numpy.zeros(5) # SDSS ugriz for Schafly  2011
#    sdss98=numpy.zeros(5) # SDSS ugriz for Schlegel 1998
#    ukirt11=numpy.zeros(4) # UKIRT JHKL for Schafly  2011
#    ukirt98=numpy.zeros(4) # UKIRT JHKL for Schlegel 1998

#    for i in range(len(filterlist)):
#        stringposition=nedlist.find(filterlist[i])
        #print nedlist[stringposition:stringposition+79]
        #print nedlist[stringposition+18:stringposition+23]
        #print nedlist[stringposition+74:stringposition+79]
#        if(i>=0 and i<=4):
#           landolt11[i]=eval(nedlist[stringposition+18:stringposition+23])
#           landolt98[i]=eval(nedlist[stringposition+74:stringposition+79])
#        elif(i>=5 and i<=9):
#           sdss11[i-5]=eval(nedlist[stringposition+18:stringposition+23])
#           sdss98[i-5]=eval(nedlist[stringposition+74:stringposition+79])
#        elif(i>=10 and i<=13):
#           ukirt11[i-10]=eval(nedlist[stringposition+18:stringposition+23])
#           ukirt98[i-10]=eval(nedlist[stringposition+74:stringposition+79])
        
#    mwebv=landolt98[1]-landolt98[2]
#    mwebv2011=landolt11[1]-landolt11[2]
#    return [mwebv,mwebv2011,landolt11,sdss11,ukirt11,landolt98,sdss98,ukirt98]

def exec_vrebinning_flux(vpix,wave,flux):      
# Nao Suzuki
# Log Linear Velocity Binning
# Input  : vpix (km/s), wave, flux
# OUtput : rwave,rflux,rid 
      c=299792.458
      npix=len(wave)
      bwave=numpy.zeros(npix+1)
      rwave=numpy.zeros(npix+1)

# Define Blue Edge
      bwave[0]=math.sqrt(wave[0]**3/wave[1])
      rwave[0]=math.sqrt(wave[0]*wave[1])
# Define Red Edge
      bwave[npix]=math.sqrt(wave[npix-2]*wave[npix-1])
      rwave[npix]=math.sqrt(wave[npix-1]**3/wave[npix-2])

# Define each pixels blue and red edges
      for i in range(1,npix-1):
         bwave[i]=math.sqrt(wave[i-1]*wave[i])
         rwave[i]=math.sqrt(wave[i]*wave[i+1])

# Search for Starting Point

# Find Blue Edge
      hmin=int(c/vpix*math.log(wave[0]/3000.0))
      blue=3000.0*math.exp(vpix/c*(float(hmin)-0.5))
      i=0
      while (blue < bwave[0]):
        i=i+1
        blue=3000.0*math.exp(vpix/c*(float(hmin)-0.5+float(i)))
      hmin=hmin+i

# Find Red Edge
      hmax=int(c/vpix*math.log(wave[-1]/3000.0))
      red=3000.0*math.exp(vpix/c*(float(hmax)+0.5))
      j=0
      while (red > rwave[-1]):
        j=j+1
        red=3000.0*math.exp(vpix/c*(float(hmax)+0.5-float(j)))
      hmax=hmax-j
      kmax=hmax-hmin+1
      #print 'hmin,hmax,kmax is', hmin, hmax, kmax

# Define Refrence Frame
      reflambda=numpy.zeros(kmax)
      refblambda=numpy.zeros(kmax)
      refrlambda=numpy.zeros(kmax)
      rid=numpy.zeros(kmax,dtype=numpy.int)
      rebinnedflux=numpy.zeros(kmax)

      for k in range(kmax):
         reflambda[k]=3000.0*math.exp(vpix/c*float(k+hmin))
         refblambda[k]=3000.0*math.exp(vpix/c*(float(k+hmin)-0.5))
         refrlambda[k]=3000.0*math.exp(vpix/c*(float(k+hmin)+0.5))
         rid[k]=k+hmin

# Rebinning Starts from here

      imin=1
      imax=npix
      #imax=min(imin+100,npix)
      fluxS=0.0

      for k in range(kmax):
         #for h in range(npix):
         for h in range(imin,imax):

#  Case Blue Side
#  ref[k]         |-----------|
#  wave[h]    |-------|
           if(bwave[h] <= refblambda[k] and \
              rwave[h] >= refblambda[k] and \
              rwave[h] <  refrlambda[k]): 
              fluxS=fluxS+flux[h]*(rwave[h]-refblambda[k])
              imax=h

#  Case Middle I
#  ref(k)         |-----------|
#  wave(h)          |-------|
           elif(refblambda[k] <= bwave[h] and \
                refrlambda[k] >= rwave[h]): 
                fluxS=fluxS+flux[h]*(rwave[h]-bwave[h])
                imax=h

#  Case Middle II
#  ref(k)         |-----------|
#  wave(h)     |----------------|

           elif(refblambda[k] >= bwave[h] and \
                refrlambda[k] <= rwave[h]):
                fluxS=fluxS+flux[h]*(refrlambda[k]-refblambda[k])
                imax=h

#  Case Red Side
#  ref(k)         |-----------|
#  wave(h)               |-------|
           elif(rwave[h] >  refrlambda[k] and \
                bwave[h] >= refblambda[k] and \
                bwave[h] <= refrlambda[k]):
                fluxS=fluxS+flux[h]*(refrlambda[k]-bwave[h])
                imax=h

         rebinnedflux[k]=fluxS/(refrlambda[k]-refblambda[k])
         fluxS=0.0
         imin=max(0,imax-50)
         imax=min(npix,imax+50)

      rnum=kmax
      return [reflambda,rebinnedflux,rid]

def exec_vrebinning_fluxivar(vpix,wave,flux,ivar):      
# Nao Suzuki
# Log Linear Velocity Binning
# Input  : vpix (km/s), wave, flux, ivar
# OUtput : rwave,rflux,rivar,rid
      c=299792.458
      npix=len(wave)
      bwave=numpy.zeros(npix+1)
      rwave=numpy.zeros(npix+1)

# Define Blue Edge
      bwave[0]=math.sqrt(wave[0]**3/wave[1])
      rwave[0]=math.sqrt(wave[0]*wave[1])
# Define Red Edge
      bwave[npix]=math.sqrt(wave[npix-2]*wave[npix-1])
      rwave[npix]=math.sqrt(wave[npix-1]**3/wave[npix-2])

# Define each pixels blue and red edges
      for i in range(1,npix-1):
         bwave[i]=math.sqrt(wave[i-1]*wave[i])
         rwave[i]=math.sqrt(wave[i]*wave[i+1])

# Search for Starting Point

# Find Blue Edge
      hmin=int(c/vpix*math.log(wave[0]/3000.0))
      blue=3000.0*math.exp(vpix/c*(float(hmin)-0.5))
      i=0
      while (blue < bwave[0]):
        i=i+1
        blue=3000.0*math.exp(vpix/c*(float(hmin)-0.5+float(i)))
      hmin=hmin+i

# Find Red Edge
      hmax=int(c/vpix*math.log(wave[-1]/3000.0))
      red=3000.0*math.exp(vpix/c*(float(hmax)+0.5))
      j=0
      while (red > rwave[-1]):
        j=j+1
        red=3000.0*math.exp(vpix/c*(float(hmax)+0.5-float(j)))
      hmax=hmax-j
      kmax=hmax-hmin+1
      #print 'hmin,hmax,kmax is', hmin, hmax, kmax

# Define Refrence Frame
      refblambda=numpy.zeros(kmax)
      refrlambda=numpy.zeros(kmax)

      reflambda=numpy.zeros(kmax)
      rebinnedflux=numpy.zeros(kmax)
      rebinnedivar=numpy.zeros(kmax)
      rid=numpy.zeros(kmax,dtype=numpy.int)

      for k in range(kmax):
         reflambda[k]=3000.0*math.exp(vpix/c*float(k+hmin))
         refblambda[k]=3000.0*math.exp(vpix/c*(float(k+hmin)-0.5))
         refrlambda[k]=3000.0*math.exp(vpix/c*(float(k+hmin)+0.5))
         rid[k]=k+hmin

# Rebinning Starts from here

      imin=1
      imax=npix
      fluxS=0.0
      ivarS=0.0

      for k in range(kmax):
         #for h in range(npix):
         for h in range(imin,imax):

#  Case Blue Side
#  ref[k]         |-----------|
#  wave[h]    |-------|
           if(bwave[h] <= refblambda[k] and \
              rwave[h] >= refblambda[k] and \
              rwave[h] <  refrlambda[k]): 
              fluxS=fluxS+flux[h]*(rwave[h]-refblambda[k])
              ivarS=ivarS+ivar[h]*(rwave[h]-refblambda[k])
              imax=h

#  Case Middle I
#  ref(k)         |-----------|
#  wave(h)          |-------|
           elif(refblambda[k] <= bwave[h] and \
                refrlambda[k] >= rwave[h]): 
                fluxS=fluxS+flux[h]*(rwave[h]-bwave[h])
                ivarS=ivarS+ivar[h]*(rwave[h]-bwave[h])
                imax=h

#  Case Middle II
#  ref(k)         |-----------|
#  wave(h)     |----------------|

           elif(refblambda[k] >= bwave[h] and \
                refrlambda[k] <= rwave[h]):
                fluxS=fluxS+flux[h]*(refrlambda[k]-refblambda[k])
                ivarS=ivarS+ivar[h]*(refrlambda[k]-refblambda[k])
                imax=h

#  Case Red Side
#  ref(k)         |-----------|
#  wave(h)               |-------|
           elif(rwave[h] >  refrlambda[k] and \
                bwave[h] >= refblambda[k] and \
                bwave[h] <= refrlambda[k]):
                fluxS=fluxS+flux[h]*(refrlambda[k]-bwave[h])
                ivarS=ivarS+ivar[h]*(refrlambda[k]-bwave[h])
                imax=h

         rebinnedflux[k]=fluxS/(refrlambda[k]-refblambda[k])
         rebinnedivar[k]=ivarS/(refrlambda[k]-refblambda[k])
         fluxS=0.0
         ivarS=0.0
         imin=max(0,imax-50)
         imax=min(npix,imax+50)

      return [reflambda,rebinnedflux,rebinnedivar,rid]

def exec_logrebinning_flux_fromlinear(dlog,wave,flux):
# Nao Suzuki
# Jun 12, 2012 (Tue) 4:56pm
# Log Linear Velocity Binning
# Input  : dlog (log bin like 0.0001), wave, flux, ivar
# OUtput : rwave,rflux,rivar,rid
      npix=len(wave)
      bwave=numpy.zeros(npix+1)
      rwave=numpy.zeros(npix+1)

# Define Blue Edge
      #bwave[0]=math.sqrt(wave[0]**3/wave[1])
      #rwave[0]=math.sqrt(wave[0]*wave[1])
      bwave[0]=(wave[0]+wave[1])/2.0
      rwave[0]=(wave[1]+wave[2])/2.0
# Define Red Edge
      #bwave[npix]=math.sqrt(wave[npix-2]*wave[npix-1])
      #rwave[npix]=math.sqrt(wave[npix-1]**3/wave[npix-2])
      bwave[npix]=(wave[npix-3]+wave[npix-2])/2.0
      rwave[npix]=(wave[npix-2]+wave[npix-1])/2.0

# Define each pixels blue and red edges
      for i in range(1,npix-1):
         #bwave[i]=math.sqrt(wave[i-1]*wave[i])
         #rwave[i]=math.sqrt(wave[i]*wave[i+1])
         bwave[i]=(wave[i-1]+wave[i])/2.0
         rwave[i]=(wave[i]+wave[i+1])/2.0

# Search for Starting Point

# Find Blue Edge
      #hmin=int(c/vpix*math.log(wave[0]/3000.0))
      #blue=3000.0*math.exp(vpix/c*(float(hmin)-0.5))

      hmin=int(math.log10(wave[0])/dlog)
      blue=10.0**((float(hmin)-0.5)*dlog)
      i=0
      while (blue < bwave[0]):
        i=i+1
        #blue=3000.0*math.exp(vpix/c*(float(hmin)-0.5+float(i)))
        blue=10.0**((float(hmin)-0.5+float(i))*dlog)
      hmin=hmin+i

# Find Red Edge
      #hmax=int(c/vpix*math.log(wave[-1]/3000.0))
      #red=3000.0*math.exp(vpix/c*(float(hmax)+0.5))

      hmax=int(math.log10(wave[-1])/dlog)
      red=10.0**((float(hmax)+0.5)*dlog)
      j=0
      while (red > rwave[-1]):
        j=j+1
        #red=3000.0*math.exp(vpix/c*(float(hmax)+0.5-float(j)))
        red=10.0**((float(hmax)+0.5-float(j))*dlog)
      hmax=hmax-j
      kmax=hmax-hmin+1
      #print 'hmin,hmax,kmax is', hmin, hmax, kmax

# Define Refrence Frame
      refblambda=numpy.zeros(kmax)
      refrlambda=numpy.zeros(kmax)

      reflambda=numpy.zeros(kmax)
      rebinnedflux=numpy.zeros(kmax)
      rid=numpy.zeros(kmax,dtype=numpy.int32)

      for k in range(kmax):
         #reflambda[k]=3000.0*math.exp(vpix/c*float(k+hmin))
         #refblambda[k]=3000.0*math.exp(vpix/c*(float(k+hmin)-0.5))
         #refrlambda[k]=3000.0*math.exp(vpix/c*(float(k+hmin)+0.5))
         reflambda[k]=10.0**(float(k+hmin)*dlog)
         refblambda[k]=10.0**((float(k+hmin)-0.5)*dlog)
         refrlambda[k]=10.0**((float(k+hmin)+0.5)*dlog)
         rid[k]=k+hmin

# Rebinning Starts from here

      #imin=1
      #imax=min(imin+50,npix)

      imin=0
      imax=npix
      #imax=min(imin+100,npix)
      fluxS=0.0

      for k in range(kmax):
         #for h in range(npix):
         for h in range(imin,imax):

#  Case Blue Side
#  ref[k]         |-----------|
#  wave[h]    |-------|
           if(bwave[h] <= refblambda[k] and \
              rwave[h] >= refblambda[k] and \
              rwave[h] <  refrlambda[k]):
              fluxS=fluxS+flux[h]*(rwave[h]-refblambda[k])
              imax=h

#  Case Middle I
#  ref(k)         |-----------|
#  wave(h)          |-------|
           elif(refblambda[k] <= bwave[h] and \
                refrlambda[k] >= rwave[h]):
                #print h,imin,imax,bwave[h],rwave[h],flux[h]
                fluxS=fluxS+flux[h]*(rwave[h]-bwave[h])
                imax=h

#  Case Middle II
#  ref(k)         |-----------|
#  wave(h)     |----------------|

           elif(refblambda[k] >= bwave[h] and \
                refrlambda[k] <= rwave[h]):
                fluxS=fluxS+flux[h]*(refrlambda[k]-refblambda[k])
                imax=h

#  Case Red Side
#  ref(k)         |-----------|
#  wave(h)               |-------|
           elif(rwave[h] >  refrlambda[k] and \
                bwave[h] >= refblambda[k] and \
                bwave[h] <= refrlambda[k]):
                fluxS=fluxS+flux[h]*(refrlambda[k]-bwave[h])
                imax=h

         rebinnedflux[k]=fluxS/(refrlambda[k]-refblambda[k])
         fluxS=0.0
         #imin=max(0,imax-50)
         #imax=min(npix,imax+50)
         imin=max(0,imax-100)
         imax=min(npix,imax+100)
         #imin=max(0,imax-500)
         #imax=min(npix-2,imax+500)

      return [reflambda,rebinnedflux,rid]

def exec_logrebinning_fluxivar(dlog,wave,flux,ivar):
# Jun 12, 2012 (Tue) 4:56pm
# Log Linear Velocity Binning
# Input  : dlog (log bin like 0.0001), wave, flux, ivar
# OUtput : rwave,rflux,rivar,rid
      npix=len(wave)
      bwave=numpy.zeros(npix+1)
      rwave=numpy.zeros(npix+1)

# Define Blue Edge
      bwave[0]=math.sqrt(wave[0]**3/wave[1])
      rwave[0]=math.sqrt(wave[0]*wave[1])
# Define Red Edge
      bwave[npix]=math.sqrt(wave[npix-2]*wave[npix-1])
      rwave[npix]=math.sqrt(wave[npix-1]**3/wave[npix-2])

# Define each pixels blue and red edges
      for i in range(1,npix-1):
         bwave[i]=math.sqrt(wave[i-1]*wave[i])
         rwave[i]=math.sqrt(wave[i]*wave[i+1])

# Search for Starting Point

# Find Blue Edge
      #hmin=int(c/vpix*math.log(wave[0]/3000.0))
      #blue=3000.0*math.exp(vpix/c*(float(hmin)-0.5))

      hmin=int(math.log10(wave[0])/dlog)
      blue=10.0**((float(hmin)-0.5)*dlog)
      i=0
      while (blue < bwave[0]):
        i=i+1
        #blue=3000.0*math.exp(vpix/c*(float(hmin)-0.5+float(i)))
        blue=10.0**((float(hmin)-0.5+float(i))*dlog)
      hmin=hmin+i

# Find Red Edge
      #hmax=int(c/vpix*math.log(wave[-1]/3000.0))
      #red=3000.0*math.exp(vpix/c*(float(hmax)+0.5))

      hmax=int(math.log10(wave[-1])/dlog)
      red=10.0**((float(hmax)+0.5)*dlog)
      j=0
      while (red > rwave[-1]):
        j=j+1
        #red=3000.0*math.exp(vpix/c*(float(hmax)+0.5-float(j)))
        red=10.0**((float(hmax)+0.5-float(j))*dlog)
      hmax=hmax-j
      kmax=hmax-hmin+1
      #print 'hmin,hmax,kmax is', hmin, hmax, kmax

# Define Refrence Frame
      refblambda=numpy.zeros(kmax)
      refrlambda=numpy.zeros(kmax)

      reflambda=numpy.zeros(kmax)
      rebinnedflux=numpy.zeros(kmax)
      rebinnedivar=numpy.zeros(kmax)
      rid=numpy.zeros(kmax,dtype=numpy.int)

      for k in range(kmax):
         #reflambda[k]=3000.0*math.exp(vpix/c*float(k+hmin))
         #refblambda[k]=3000.0*math.exp(vpix/c*(float(k+hmin)-0.5))
         #refrlambda[k]=3000.0*math.exp(vpix/c*(float(k+hmin)+0.5))
         reflambda[k]=10.0**(float(k+hmin)*dlog)
         refblambda[k]=10.0**((float(k+hmin)-0.5)*dlog)
         refrlambda[k]=10.0**((float(k+hmin)+0.5)*dlog)
         rid[k]=k+hmin

# Rebinning Starts from here

      imin=1
      imax=min(imin+50,npix)

      #imin=0
      #imax=npix
      #imax=min(imin+100,npix)
      fluxS=0.0
      ivarS=0.0

      for k in range(kmax):
         #for h in range(npix):
         for h in range(imin,imax):

#  Case Blue Side
#  ref[k]         |-----------|
#  wave[h]    |-------|
           if(bwave[h] <= refblambda[k] and \
              rwave[h] >= refblambda[k] and \
              rwave[h] <  refrlambda[k]):
              fluxS=fluxS+flux[h]*(rwave[h]-refblambda[k])
              ivarS=ivarS+ivar[h]*(rwave[h]-refblambda[k])
              imax=h

#  Case Middle I
#  ref(k)         |-----------|
#  wave(h)          |-------|
           elif(refblambda[k] <= bwave[h] and \
                refrlambda[k] >= rwave[h]):
                fluxS=fluxS+flux[h]*(rwave[h]-bwave[h])
                ivarS=ivarS+ivar[h]*(rwave[h]-bwave[h])
                imax=h

#  Case Middle II
#  ref(k)         |-----------|
#  wave(h)     |----------------|

           elif(refblambda[k] >= bwave[h] and \
                refrlambda[k] <= rwave[h]):
                fluxS=fluxS+flux[h]*(refrlambda[k]-refblambda[k])
                ivarS=ivarS+ivar[h]*(refrlambda[k]-refblambda[k])
                imax=h

#  Case Red Side
#  ref(k)         |-----------|
#  wave(h)               |-------|
           elif(rwave[h] >  refrlambda[k] and \
                bwave[h] >= refblambda[k] and \
                bwave[h] <= refrlambda[k]):
                fluxS=fluxS+flux[h]*(refrlambda[k]-bwave[h])
                ivarS=ivarS+ivar[h]*(refrlambda[k]-bwave[h])
                imax=h

         rebinnedflux[k]=fluxS/(refrlambda[k]-refblambda[k])
         rebinnedivar[k]=ivarS/(refrlambda[k]-refblambda[k])
         fluxS=0.0
         ivarS=0.0
         imin=max(0,imax-50)
         imax=min(npix,imax+50)

      return [reflambda,rebinnedflux,rebinnedivar,rid]

def exec_logrebinning_fluxivarmask(dlog,wave,flux,ivar,mask):
# Nao Suzuki
# Jun 12, 2012 (Tue) 7:32pm
# Log Linear Binning
# Input  : dlog (0.0001), wave, flux, ivar, mask
# Output : rwave,rflux,rivar,rid

      npix=len(wave)
      bwave=numpy.zeros(npix+1)
      rwave=numpy.zeros(npix+1)

# Define Blue Edge
      bwave[0]=math.sqrt(wave[0]**3/wave[1])
      rwave[0]=math.sqrt(wave[0]*wave[1])
# Define Red Edge
      bwave[npix]=math.sqrt(wave[npix-2]*wave[npix-1])
      rwave[npix]=math.sqrt(wave[npix-1]**3/wave[npix-2])

# Define each pixels blue and red edges
      for i in range(1,npix-1):
         bwave[i]=math.sqrt(wave[i-1]*wave[i])
         rwave[i]=math.sqrt(wave[i]*wave[i+1])

# Search for Starting Point

# Find Blue Edge
      hmin=int(math.log10(wave[0])/dlog)
      blue=10.0**((float(hmin)-0.5)*dlog)
      i=0
      while (blue < bwave[0]):
        i=i+1
        blue=10.0**((float(hmin)-0.5+float(i))*dlog)
      hmin=hmin+i

# Find Red Edge
      hmax=int(math.log10(wave[-1])/dlog)
      red=10.0**((float(hmax)+0.5)*dlog)
      j=0
      while (red > rwave[-1]):
        j=j+1
        red=10.0**((float(hmax)+0.5-float(j))*dlog)
      hmax=hmax-j
      kmax=hmax-hmin+1

# Define Refrence Frame
      refblambda=numpy.zeros(kmax)
      refrlambda=numpy.zeros(kmax)

      reflambda=numpy.zeros(kmax)
      rebinnedflux=numpy.zeros(kmax)
      rebinnedivar=numpy.zeros(kmax)
      rid=numpy.zeros(kmax,dtype=numpy.int)
      rebinnedmask=numpy.zeros(kmax,dtype=numpy.int)

      for k in range(kmax):
         reflambda[k]=10.0**(float(k+hmin)*dlog)
         refblambda[k]=10.0**((float(k+hmin)-0.5)*dlog)
         refrlambda[k]=10.0**((float(k+hmin)+0.5)*dlog)
         rid[k]=k+hmin

# Rebinning Starts from here

      imin=1
      imax=npix
      imax=min(imin+50,npix)
      fluxS=0.0
      ivarS=0.0
      maskS=0

      for k in range(kmax):
         for h in range(imin,imax):

#  Case Blue Side
#  ref[k]         |-----------|
#  wave[h]    |-------|
           if(bwave[h] <= refblambda[k] and \
              rwave[h] >= refblambda[k] and \
              rwave[h] <  refrlambda[k]):
              fluxS=fluxS+flux[h]*(rwave[h]-refblambda[k])
              ivarS=ivarS+ivar[h]*(rwave[h]-refblambda[k])
              imax=h
              if(mask[h]!=0): maskS=max(mask[h],maskS)

#  Case Middle I
#  ref(k)         |-----------|
#  wave(h)          |-------|
           elif(refblambda[k] <= bwave[h] and \
                refrlambda[k] >= rwave[h]):
                fluxS=fluxS+flux[h]*(rwave[h]-bwave[h])
                ivarS=ivarS+ivar[h]*(rwave[h]-bwave[h])
                imax=h
                if(mask[h]!=0): maskS=max(mask[h],maskS)

#  Case Middle II
#  ref(k)         |-----------|
#  wave(h)     |----------------|

           elif(refblambda[k] >= bwave[h] and \
                refrlambda[k] <= rwave[h]):
                fluxS=fluxS+flux[h]*(refrlambda[k]-refblambda[k])
                ivarS=ivarS+ivar[h]*(refrlambda[k]-refblambda[k])
                imax=h
                if(mask[h]!=0): maskS=max(mask[h],maskS)

#  Case Red Side
#  ref(k)         |-----------|
#  wave(h)               |-------|
           elif(rwave[h] >  refrlambda[k] and \
                bwave[h] >= refblambda[k] and \
                bwave[h] <= refrlambda[k]):
                fluxS=fluxS+flux[h]*(refrlambda[k]-bwave[h])
                ivarS=ivarS+ivar[h]*(refrlambda[k]-bwave[h])
                imax=h
                if(mask[h]!=0): maskS=max(mask[h],maskS)

         rebinnedflux[k]=fluxS/(refrlambda[k]-refblambda[k])
         rebinnedivar[k]=ivarS/(refrlambda[k]-refblambda[k])
         rebinnedmask[k]=maskS
         fluxS=0.0
         ivarS=0.0
         maskS=0
         imin=max(0,imax-50)
         imax=min(npix,imax+50)

      return [reflambda,rebinnedflux,rebinnedivar,rid,rebinnedmask]

def exec_logrebinning_fluxivarmask2(dlog,wave,flux,ivar,mask,lambdaflux,lambdaivar):
# Nao Suzuki
# Jun 12, 2012 (Tue) 7:32pm
# Log Linear Binning
# Input  : dlog (0.0001), wave, flux, ivar, mask, lambdaflux, lambdaivar
# Output : rwave,rflux,rivar,rid,mask,lambdaflux,lambdaiva

      npix=len(wave)
      bwave=numpy.zeros(npix+1)
      rwave=numpy.zeros(npix+1)

# Define Blue Edge
      bwave[0]=math.sqrt(wave[0]**3/wave[1])
      rwave[0]=math.sqrt(wave[0]*wave[1])
# Define Red Edge
      bwave[npix]=math.sqrt(wave[npix-2]*wave[npix-1])
      rwave[npix]=math.sqrt(wave[npix-1]**3/wave[npix-2])

# Define each pixels blue and red edges
      for i in range(1,npix-1):
         bwave[i]=math.sqrt(wave[i-1]*wave[i])
         rwave[i]=math.sqrt(wave[i]*wave[i+1])

# Search for Starting Point

# Find Blue Edge
      hmin=int(math.log10(wave[0])/dlog)
      blue=10.0**((float(hmin)-0.5)*dlog)
      i=0
      while (blue < bwave[0]):
        i=i+1
        blue=10.0**((float(hmin)-0.5+float(i))*dlog)
      hmin=hmin+i

# Find Red Edge
      hmax=int(math.log10(wave[-1])/dlog)
      red=10.0**((float(hmax)+0.5)*dlog)
      j=0
      while (red > rwave[-1]):
        j=j+1
        red=10.0**((float(hmax)+0.5-float(j))*dlog)
      hmax=hmax-j
      kmax=hmax-hmin+1

# Define Refrence Frame
      refblambda=numpy.zeros(kmax)
      refrlambda=numpy.zeros(kmax)

      reflambda=numpy.zeros(kmax)
      rebinnedflux=numpy.zeros(kmax)
      rebinnedivar=numpy.zeros(kmax)
      rid=numpy.zeros(kmax,dtype=numpy.int)
      rebinnedmask=numpy.zeros(kmax,dtype=numpy.int)

      rebinnedlambdaflux=numpy.zeros(kmax)
      rebinnedlambdaivar=numpy.zeros(kmax)

      for k in range(kmax):
         reflambda[k]=10.0**(float(k+hmin)*dlog)
         refblambda[k]=10.0**((float(k+hmin)-0.5)*dlog)
         refrlambda[k]=10.0**((float(k+hmin)+0.5)*dlog)
         rid[k]=k+hmin

# Rebinning Starts from here

      imin=1
      #imax=npix
      imax=min(imin+50,npix)
      fluxS=0.0
      ivarS=0.0
      maskS=0

      lambdafluxS=0.0
      lambdaivarS=0.0

      for k in range(kmax):
         #print 'kth wavelength bin',refblambda[k],reflambda[k],refrlambda[k]
         for h in range(imin,imax):

#  Case Blue Side
#  ref[k]         |-----------|
#  wave[h]    |-------|
           if(bwave[h] <= refblambda[k] and \
              rwave[h] >= refblambda[k] and \
              rwave[h] <  refrlambda[k]):
              fluxS=fluxS+flux[h]*(rwave[h]-refblambda[k])
              ivarS=ivarS+ivar[h]*(rwave[h]-refblambda[k])
              imax=h
              lambdafluxS=fluxS+lambdaflux[h]*(rwave[h]-refblambda[k])
              lambdaivarS=ivarS+lambdaivar[h]*(rwave[h]-refblambda[k])
              if(mask[h]!=0): maskS=max(mask[h],maskS)

#  Case Middle I
#  ref(k)         |-----------|
#  wave(h)          |-------|
           elif(refblambda[k] <= bwave[h] and \
                refrlambda[k] >= rwave[h]):
                fluxS=fluxS+flux[h]*(rwave[h]-bwave[h])
                ivarS=ivarS+ivar[h]*(rwave[h]-bwave[h])
                imax=h
                lambdafluxS=fluxS+lambdaflux[h]*(rwave[h]-bwave[h])
                lambdaivarS=ivarS+lambdaivar[h]*(rwave[h]-bwave[h])
                if(mask[h]!=0): maskS=max(mask[h],maskS)

#  Case Middle II
#  ref(k)         |-----------|
#  wave(h)     |----------------|

           elif(refblambda[k] >= bwave[h] and \
                refrlambda[k] <= rwave[h]):
                fluxS=fluxS+flux[h]*(refrlambda[k]-refblambda[k])
                ivarS=ivarS+ivar[h]*(refrlambda[k]-refblambda[k])
                imax=h
                lambdafluxS=fluxS+lambdaflux[h]*(refrlambda[k]-refblambda[k])
                lambdaivarS=ivarS+lambdaivar[h]*(refrlambda[k]-refblambda[k])
                if(mask[h]!=0): maskS=max(mask[h],maskS)

#  Case Red Side
#  ref(k)         |-----------|
#  wave(h)               |-------|
           elif(rwave[h] >  refrlambda[k] and \
                bwave[h] >= refblambda[k] and \
                bwave[h] <= refrlambda[k]):
                fluxS=fluxS+flux[h]*(refrlambda[k]-bwave[h])
                ivarS=ivarS+ivar[h]*(refrlambda[k]-bwave[h])
                imax=h
                lambdafluxS=fluxS+lambdaflux[h]*(refrlambda[k]-bwave[h])
                lambdaivarS=ivarS+lambdaivar[h]*(refrlambda[k]-bwave[h])
                if(mask[h]!=0): maskS=max(mask[h],maskS)

         rebinnedflux[k]=fluxS/(refrlambda[k]-refblambda[k])
         rebinnedivar[k]=ivarS/(refrlambda[k]-refblambda[k])
         rebinnedmask[k]=maskS
         rebinnedlambdaflux[k]=lambdafluxS/(refrlambda[k]-refblambda[k])
         rebinnedlambdaivar[k]=lambdaivarS/(refrlambda[k]-refblambda[k])
         fluxS=0.0
         ivarS=0.0
         maskS=0
         lambdafluxS=0.0
         lambdaivarS=0.0
         imin=max(0,imax-50)
         imax=min(npix,imax+50)

      return [reflambda,rebinnedflux,rebinnedivar,rid,rebinnedmask,rebinnedlambdaflux,rebinnedlambdaivar]


