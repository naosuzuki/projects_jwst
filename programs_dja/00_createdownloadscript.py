import pandas as pd
import sys

def djadownload():
   df=pd.read_csv('../csvfiles/dja_20250309.csv')
   print(df)
   for i in range(len(df)):
      uid=df["uid"].iloc[i]
      fitsfile=df["file"].iloc[i]
      rootname=fitsfile.split('_')[0]
      #print(i,uid,rootname,fitsfile)
      print("wget "+"https://s3.amazonaws.com/msaexp-nirspec/extractions/"+rootname+"/"+fitsfile+"\n")
djadownload()
