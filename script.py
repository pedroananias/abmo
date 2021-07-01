#!/usr/bin/python
# -*- coding: utf-8 -*-

#########################################################################################################################################
# ### ABMO - Algal Bloom Monthly Occurences
# ### Script responsible for executing Algal Bloom Monthly Occurences in a region of interest
# ### Python 3.7 64Bits is required!
#########################################################################################################################################

# ### Version
version = "V4"


# ### Module imports

# Main
import ee
import time
import os
import sys
import argparse
import traceback
import pandas as pd

# Sub
from datetime import datetime as dt

# Extras modules
from modules import misc, gee, abmo


# ### Script args parsing

# starting arg parser
parser = argparse.ArgumentParser(description=version)

# create arguments
parser.add_argument('--lat_lon', dest='lat_lon', action='store', default="-48.56620427006758,-22.457495449468666,-47.9777042099919,-22.80261692472655",
                   help="Two diagnal points (Latitude 1, Longitude 1, Latitude 2, Longitude 2) of the study area")
parser.add_argument('--date_start', dest='date_start', action='store', default="1985-01-01",
                   help="Date to start time series")
parser.add_argument('--date_end', dest='date_end', action='store', default="2001-12-31",
                   help="Date to end time series")
parser.add_argument('--date_start2', dest='date_start2', action='store', default=None,
                   help="Date to start time series (second period)")
parser.add_argument('--date_end2', dest='date_end2', action='store', default=None,
                   help="Date to end time series (second period)")
parser.add_argument('--name', dest='name', action='store', default="bbhr",
                   help="Place where to save generated files")
parser.add_argument('--sensor', dest='sensor', action='store', default="landsat578",
                   help="Define which sensor will be used")
parser.add_argument('--indice', dest='indice', action='store', default="mndwi,ndvi,fai,sabi,slope",
                   help="Define which indice will be used to determine algal blooms (mndwi, ndvi, fai, sabi e/ou slope)")
parser.add_argument('--min_occurrence', dest='min_occurrence', type=int, action='store', default=4,
                   help="Define how many indices will have to match in order to determine pixel as algal bloom occurrence")
parser.add_argument('--seasonal', dest='seasonal', action='store_true',
                   help="Define if pixels will be reduced by using seasons instead of months")
parser.add_argument('--shapefile', dest='shapefile', action='store',
                   help="Use a shapefile to clip a region of interest")
parser.add_argument('--force_cache', dest='force_cache', action='store_true',
                   help="Force cache reseting to prevent image errors")

# parsing arguments
args = parser.parse_args()




# ### Start

try:

  # Start script time counter
  start_time = time.time()

  # Google Earth Engine API initialization
  ee.Initialize()



  # ### Working directory

  # Data path
  folderRoot = os.path.dirname(os.path.realpath(__file__))+'/data'
  if not os.path.exists(folderRoot):
    os.mkdir(folderRoot)

  # Images path
  folderCache = os.path.dirname(os.path.realpath(__file__))+'/cache'
  if not os.path.exists(folderCache):
    os.mkdir(folderCache)


  
  # ### ABMO execution

  # folder to save results from algorithm at
  folder = folderRoot+'/'+dt.now().strftime("%Y%m%d_%H%M%S")+'[v='+str(version)+'-'+str(args.name)+',dstart='+str(args.date_start)+',dend='+str(args.date_end)+',dstart2='+str(args.date_start2)+',dend2='+str(args.date_end2)+',i='+str(args.indice)+',moc='+str(args.min_occurrence)+',s='+str(args.seasonal)+']'
  if not os.path.exists(folder):
    os.mkdir(folder)

  # create algorithm
  abmo = abmo.Abmo(lat_lon=args.lat_lon,
                   date_start=dt.strptime(args.date_start, "%Y-%m-%d"),
                   date_end=dt.strptime(args.date_end, "%Y-%m-%d"),
                   date_start2=dt.strptime(args.date_start2, "%Y-%m-%d") if not args.date_start2 is None else None,
                   date_end2=dt.strptime(args.date_end2, "%Y-%m-%d") if not args.date_end2 is None else None,
                   sensor=args.sensor,
                   cache_path=folderCache, 
                   force_cache=args.force_cache,
                   indice=args.indice,
                   min_occurrence=args.min_occurrence,
                   seasonal=args.seasonal,
                   shapefile=args.shapefile)

  # preprocessing
  abmo.process_timeseries_data()

  # create plot
  abmo.save_occurrences_plot(df=abmo.df_timeseries, folder=folder)

  # save timeseries in csv file
  abmo.save_dataset(df=abmo.df_timeseries, path=folder+'/timeseries[dstart='+str(args.date_start)+',dend='+str(args.date_end)+',moc='+str(args.min_occurrence)+',s='+str(args.seasonal)+'].csv')

  # save geojson occurrences and clouds
  abmo.save_occurrences_geojson(df=abmo.df_timeseries, path=folder+'/occurrences[dstart='+str(args.date_start)+',dend='+str(args.date_end)+',moc='+str(args.min_occurrence)+',s='+str(args.seasonal)+'].json')

  # save images to Local Folder (first try, based on image size) or to your Google Drive
  #abmo.save_collection_tiff(folder=folder+"/tiff", folderName=args.name+"_"+version, rgb=False).

  # results
  # add results and save it on disk
  abmo.df_timeseries = abmo.df_timeseries.drop(['label'], axis=1)
  path_df_timeseries = folderRoot+'/results[moc='+str(args.min_occurrence)+',s='+str(args.seasonal)+'].csv'
  df_timeseries = pd.read_csv(path_df_timeseries).drop(['Unnamed: 0'], axis=1, errors="ignore").append(abmo.df_timeseries) if os.path.exists(path_df_timeseries) else abmo.df_timeseries.copy(deep=True)
  df_timeseries.to_csv(r''+path_df_timeseries)

  # ### Script termination notice
  script_time_all = time.time() - start_time
  debug = "***** Script execution completed successfully (-- %s seconds --) *****" %(script_time_all)
  print()
  print(debug)

except:

    # ### Script execution error warning

    # Execution
    print()
    print()
    debug = "***** Error on script execution: "+str(traceback.format_exc())
    print(debug)

    # Removes the folder created initially with the result of execution
    script_time_all = time.time() - start_time
    debug = "***** Script execution could not be completed (-- %s seconds --) *****" %(script_time_all)
    print(debug)
