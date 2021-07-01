#!/usr/bin/python
# -*- coding: utf-8 -*-

#########################################################################################################################################
# ### ABMO - Algal Bloom Monthly Occurences
# ### Module responsible for extracting Algal Bloom Monthly Occurences in a region of interest
#########################################################################################################################################

# Dependencies
# Base
import ee
import numpy as np
import pandas as pd
import hashlib
import PIL
import requests
import os
import joblib
import gc
import sys
import traceback
import math
import os
import random
import copy
import re
import time
import warnings
import geojson
from io import BytesIO
from datetime import datetime as dt
from datetime import timedelta as td
from dateutil.relativedelta import relativedelta as rd

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from matplotlib.patches import Rectangle
from six import string_types

# Local
from modules import misc, gee

# Warning correction from Pandas
pd.plotting.register_matplotlib_converters()
warnings.filterwarnings("ignore")

# Algal Bloom Monthly Occurences
class Abmo:

  # configuration
  anomaly                     = 1
  dummy                       = -99999
  max_tile_pixels             = 10000000 # if higher, will split the geometry into tiles

  # supports
  dates_timeseries            = [None, None]
  dates_timeseries_interval   = []
  sensor_params               = None

  # sample variables
  sample_total_pixel          = None
  sample_clip                 = None
  sample_lon_lat              = [[0,0],[0,0]]
  splitted_geometry           = []
  months_list                 = None
  seasons_list                = dict({1: "summer", 2: "summer", 3: "summer", 4: "autumn", 5: "autumn", 6: "autumn", 7: "winter", 8: "winter", 9: "winter", 10: "spring", 11: "spring", 12: "spring"})
  seasons_list_               = ['summer', 'autumn', 'winter', 'spring']

  # masks
  water_mask                  = None
  
  # collections
  collection                  = None
  collection_water            = None
  collection_yearly           = None

  # attributes used in timeseries
  attributes                  = ['cloud', 'label', 'occurrence', 'not_occurrence']

  # dataframes
  df_columns                  = ['pixel', 'index', 'year', 'month', 'season', 'lat', 'lon']+attributes
  df_timeseries               = None

  # hash
  hash_string                 = "abmo-20210611"

  # constructor
  def __init__(self,
               lat_lon:           str,
               date_start:        dt,
               date_end:          dt,
               sensor:            str           = "landsat578",
               cache_path:        str           = None, 
               force_cache:       bool          = False,
               morph_op:          str           = None,
               morph_op_iters:    int           = 1,
               indice:            str           = "mndwi,ndvi,fai,sabi,slope",
               min_occurrence:    int           = 4,
               seasonal:          bool          = False,
               shapefile:         str           = None):
    
    # get sensor parameters
    self.sensor_params  = gee.get_sensor_params(sensor)

    # warning
    print()
    print("Selected sensor: "+self.sensor_params['name'])

    # user defined parameters
    self.geometry                     = gee.get_geometry_from_lat_lon(lat_lon)
    self.lat_lon                      = lat_lon
    self.date_start                   = date_start
    self.date_end                     = date_end
    self.sensor                       = sensor
    self.cache_path                   = cache_path
    self.force_cache                  = force_cache
    self.morph_op                     = morph_op
    self.morph_op_iters               = morph_op_iters
    self.seasonal                     = seasonal
    self.shapefile_url                = shapefile
    self.shapefile                    = ee.FeatureCollection(self.shapefile_url) if self.shapefile_url else None

    # change GEE indice selected
    gee.indice_selected               = indice
    gee.min_occurrence                = min_occurrence

    # creating final sensor collection
    collection, collection_water      = gee.get_sensor_collections(geometry=self.geometry, sensor=self.sensor, dates=[dt.strftime(self.date_start, "%Y-%m-%d"), dt.strftime(self.date_end, "%Y-%m-%d")])

    # check if there is imags to use
    if collection.size().getInfo() > 0:

      # correc time series date start and end
      self.dates_timeseries[0]          = dt.fromtimestamp(collection.filterBounds(self.geometry).sort('system:time_start', True).first().get('system:time_start').getInfo()/1000.0)
      self.dates_timeseries[1]          = dt.fromtimestamp(collection.filterBounds(self.geometry).sort('system:time_start', False).first().get('system:time_start').getInfo()/1000.0)

      # create useful time series
      # create useful time series
      if self.shapefile:
        self.collection = collection.map(lambda image: image.clip(self.shapefile))
      else:
        self.collection = collection
      self.collection_water             = collection_water
      self.dates_timeseries_interval    = misc.remove_duplicated_dates([dt.fromtimestamp(d/1000.0).replace(hour=00, minute=00, second=00) for d in self.collection.aggregate_array("system:time_start").getInfo()])

      # build yearly collection for label band
      self.months_list                  = ee.List.sequence(0,((self.dates_timeseries[1].year - self.dates_timeseries[0].year) * 12 + (self.dates_timeseries[1].month - self.dates_timeseries[0].month)),1)
      self.months_list                  = self.months_list.map(lambda m: ee.Date(self.dates_timeseries[0].strftime("%Y-%m-%d")).advance(m,'month'))
      seasons_list_ee                   = ee.List(list(self.seasons_list.values()))
      self.collection                   = self.collection.map(lambda i: i.set('season', ee.String(ee.Date(i.get('system:time_start')).get('year').format()).cat("-").cat(seasons_list_ee.get(ee.Date(i.get('system:time_start')).get('month').subtract(1)))))
      if self.seasonal:
        self.collection_yearly = ee.ImageCollection.fromImages(self.collection.aggregate_array('season').distinct().map(lambda s: self.collection.filterMetadata('season', 'equals', ee.String(s)).sum().set('year', ee.Number.parse(ee.String(s).split("-").get(0))).set('month', seasons_list_ee.indexOf(ee.String(s).split("-").get(1)).add(1)).set('season', ee.String(s).split("-").get(1))))
      else:
        self.collection_yearly = ee.ImageCollection.fromImages(self.months_list.map(lambda m: self.collection.filter(ee.Filter.calendarRange(ee.Date(m).get('year'), ee.Date(m).get('year'),'year')).filter(ee.Filter.calendarRange(ee.Date(m).get('month'), ee.Date(m).get('month'),'month')).sum().set('year', ee.Date(m).get('year')).set('month', ee.Date(m).get('month')).set('season', seasons_list_ee.get(ee.Date(m).get('month').subtract(1)))))

      # get monthly dates range
      self.months_list                  = [(self.dates_timeseries[1] - rd(months=int(d))) for d in range(0, self.months_list.size().getInfo())]
      self.months_list.sort()

      # preprocessing - water mask extraction
      self.water_mask                   = self.create_water_mask(self.morph_op, self.morph_op_iters)

      # count sample pixels and get sample min max coordinates
      self.sample_clip                  = self.clip_image(ee.Image(abs(self.dummy)))
      self.sample_total_pixel           = gee.get_image_counters(image=self.sample_clip.select("constant"), geometry=self.geometry, scale=self.sensor_params['scale'])["constant"]
      coordinates_min, coordinates_max  = gee.get_image_min_max(image=self.sample_clip, geometry=self.geometry, scale=self.sensor_params['scale'])
      self.sample_lon_lat               = [[float(coordinates_min['latitude']),float(coordinates_min['longitude'])],[float(coordinates_max['latitude']),float(coordinates_max['longitude'])]]

      # split geometry in tiles
      self.splitted_geometry            = self.split_geometry()

      # warning
      print("Statistics: scale="+str(self.sensor_params['scale'])+" meters, pixels="+str(self.sample_total_pixel)+", date_start='"+self.dates_timeseries[0].strftime("%Y-%m-%d")+"', date_end='"+self.dates_timeseries[1].strftime("%Y-%m-%d")+"', tiles='"+str(len(self.splitted_geometry))+"', interval_images='"+str(self.collection.size().getInfo())+"', interval_unique_images='"+str(len(self.dates_timeseries_interval))+"', yearly_images='"+str(self.collection_yearly.size().getInfo())+"', water_mask_images='"+str(self.collection_water.size().getInfo())+"', morph_op='"+str(self.morph_op)+"', morph_op_iters='"+str(self.morph_op_iters)+"'")

    # error, no images found
    else:

      # warning
      print("Error: no images were found within the selected period: date_start='"+self.date_start.strftime("%Y-%m-%d")+"', date_end='"+self.date_end.strftime("%Y-%m-%d")+"'!")
      sys.exit()

    # gargage collect
    gc.collect()


  # create the water mask
  def create_water_mask(self, morph_op: str = None, morph_op_iters: int = 1):

    # water mask
    if self.sensor == "modis":
      water_mask = self.collection_water.mode().select('water_mask').eq(1)
    elif "landsat" in self.sensor:
      water_mask = self.collection_water.mode().select('water').eq(2)
    else:
      water_mask = self.collection_water.mode().select('water').gt(0)

    # morphological operations
    if not morph_op is None and morph_op != '':
      if morph_op   == 'closing':
        water_mask = water_mask.focal_max(kernel=ee.Kernel.square(radius=1), iterations=morph_op_iters).focal_min(kernel=ee.Kernel.circle(radius=1), iterations=morph_op_iters)
      elif morph_op == 'opening':
        water_mask = water_mask.focal_min(kernel=ee.Kernel.square(radius=1), iterations=morph_op_iters).focal_max(kernel=ee.Kernel.circle(radius=1), iterations=morph_op_iters)
      elif morph_op == 'dilation':
        water_mask = water_mask.focal_max(kernel=ee.Kernel.square(radius=1), iterations=morph_op_iters)
      elif morph_op == 'erosion':
        water_mask = water_mask.focal_min(kernel=ee.Kernel.square(radius=1), iterations=morph_op_iters)

    # build image with mask
    return ee.Image(0).blend(ee.Image(abs(self.dummy)).updateMask(water_mask))


  # clipping image
  def clip_image(self, image: ee.Image, scale: int = None, geometry: ee.Geometry = None):
    geometry = self.geometry if geometry is None else geometry
    scale = self.sensor_params['scale'] if scale is None else scale
    return image.clipToBoundsAndScale(geometry=geometry, scale=scale)


  # applying water mask to indices
  def apply_water_mask(self, image: ee.Image, remove_empty_pixels = False):
    for indice in self.attributes:
        image = gee.apply_mask(image, self.water_mask, indice,  indice+"_water", remove_empty_pixels)
    return image


  # extract image from collection
  def extract_image_from_collection(self, date):
    collection = self.collection.filter(ee.Filter.date(date.strftime("%Y-%m-%d"), (date + td(days=1)).strftime("%Y-%m-%d")))
    if int(collection.size().getInfo()) == 0:
      collection = self.collection.filter(ee.Filter.date(date.strftime("%Y-%m-%d"), (date + td(days=2)).strftime("%Y-%m-%d")))
      if int(collection.size().getInfo()) == 0:
        return None
    return self.apply_water_mask(ee.Image(collection.max()).set('system:id', collection.first().get('system:id').getInfo()), False)
    

  # extract image from monthly collection
  def extract_image_from_collection_monthly(self, month):
    if self.seasonal:
      collection = self.collection_yearly.filter(ee.Filter.eq('year', month.year)).filter(ee.Filter.eq('season', self.seasons_list[month.month]))
    else:
      collection = self.collection_yearly.filter(ee.Filter.eq('year', month.year)).filter(ee.Filter.eq('month', month.month))
    if int(collection.size().getInfo()) == 0:
      return None
    return self.apply_water_mask(ee.Image(collection.first()), False)


  # split images into tiles
  def split_geometry(self):

    # check total of pixels
    total = self.sample_total_pixel*(len(self.attributes)+2)
    if total > self.max_tile_pixels:

      # total of tiles needed
      tiles = math.ceil(self.sample_total_pixel/self.max_tile_pixels)

      # lat and lons range
      latitudes       = np.linspace(self.sample_lon_lat[0][1], self.sample_lon_lat[1][1], num=tiles+1)
      longitudes      = np.linspace(self.sample_lon_lat[0][0], self.sample_lon_lat[1][0], num=tiles+1)

      # go through all latitudes and longitudes
      geometries = []
      for i, latitude in enumerate(latitudes[:-1]):
        for j, longitude in enumerate(longitudes[:-1]):
          x1 = [i,j]
          x2 = [i+1,j+1]
          geometry = gee.get_geometry_from_lat_lon(str(latitudes[x1[0]])+","+str(longitudes[x1[1]])+","+str(latitudes[x2[0]])+","+str(longitudes[x2[1]]))
          geometries.append(geometry)

      # return all created geometries
      return geometries

    else:
      
      # return single geometry
      return [gee.get_geometry_from_lat_lon(self.lat_lon)]


  # get cache files for datte
  def get_cache_files(self, month):
    prefix            = self.hash_string.encode()+str(str(self.date_start)+str(self.date_end)+str(self.date_start2)+str(self.date_end2)).encode()+self.lat_lon.encode()+self.sensor.encode()+str(self.morph_op).encode()+str(self.morph_op_iters).encode()+str(gee.indice_selected).encode()+str(gee.min_occurrence).encode()+str(self.seasonal).encode()
    hash_image        = hashlib.md5(prefix+(str(month.strftime("%Y-%m"))+'original').encode())
    hash_timeseries   = hashlib.md5(prefix+(str(self.months_list[0].strftime("%Y-%m"))+str(self.months_list[-1].strftime("%Y-%m"))).encode())
    return [self.cache_path+'/'+hash_image.hexdigest(), self.cache_path+'/'+hash_timeseries.hexdigest()]


  # process a timeseries
  def process_timeseries_data(self, force_cache: bool = False):

    # warning
    print()
    print("Starting time series processing ...")

    # attributes
    df_columns = self.df_columns+['pct_occurrence', 'pct_cloud', 'instants']
    df_timeseries  = pd.DataFrame(columns=df_columns)

    # check timeseries is already on cache
    cache_files    = self.get_cache_files(month=dt.now())
    try:

      # warning
      print("Trying to extract it from the cache...")

      # warning 2
      if self.force_cache or force_cache:
        print("User selected option 'force_cache'! Forcing building of time series...")
        raise Exception()

      # extract dataframe from cache
      df_timeseries = joblib.load(cache_files[1])
   
    # if do not exist, process normally and save it in the end
    except:

      # warning
      print("Error trying to get it from cache: either doesn't exist or is corrupted! Creating it again...")

      # process all years in time series
      if self.seasonal:
        for year in self.collection_yearly.aggregate_array('year').distinct().getInfo():
          for season in self.collection_yearly.aggregate_array('season').distinct().getInfo():
            month = dt.strptime(str(year)+"-"+str(list(self.seasons_list.values()).index(season)+1)+"-"+str(1), '%Y-%m-%d')

            # extract pixels from image
            # check if is good image (with pixels)
            df_timeseries_ = self.extract_image_pixels(image=self.extract_image_from_collection_monthly(month=month), month=month)
            if df_timeseries_.size > 0:
              df_timeseries = self.merge_timeseries(df_list=[df_timeseries, df_timeseries_])
      else:
        for month in self.months_list:

          # extract pixels from image
          # check if is good image (with pixels)
          df_timeseries_ = self.extract_image_pixels(image=self.extract_image_from_collection_monthly(month=month), month=month)
          if df_timeseries_.size > 0:
            df_timeseries = self.merge_timeseries(df_list=[df_timeseries, df_timeseries_])

      # get only good months
      # fix dataframe index
      if not df_timeseries is None:
        df_timeseries['index'] = range(0,len(df_timeseries))

        # save in cache
        if self.cache_path:
          joblib.dump(df_timeseries, cache_files[1])

    # correct columns types
    df_timeseries[['pixel','year','month']] = df_timeseries[['pixel','year','month']].astype('int64')
    df_timeseries[self.attributes+['lat','lon']] = df_timeseries[self.attributes+['lat','lon']].astype('float64')

    # remove dummies
    for attribute in [a for a in self.attributes if a != 'cloud']:
      df_timeseries = df_timeseries[(df_timeseries[attribute]!=abs(self.dummy))]

    # change cloud values
    df_timeseries.loc[df_timeseries['cloud'] == abs(self.dummy), 'cloud'] = 0.0

    # remove duplicated values
    df_timeseries.drop_duplicates(subset=['pixel','year','month','lat','lon']+self.attributes, keep='last', inplace=True)

    # # check if it is seasonal reduction
    # if self.seasonal == True:
    #   df_timeseries = df_timeseries.groupby(['year','season','lat','lon']).sum().reset_index()

    # add porcentage of occurrence and cloud
    df_timeseries['pct_occurrence']   = (df_timeseries['occurrence']/(df_timeseries['occurrence']+df_timeseries['not_occurrence']))*100
    df_timeseries['pct_cloud']        = (df_timeseries['cloud']/(df_timeseries['occurrence']+df_timeseries['not_occurrence']+df_timeseries['cloud']))*100
    df_timeseries['instants']         = df_timeseries['occurrence']+df_timeseries['not_occurrence']+df_timeseries['cloud']

    # convert to int
    df_timeseries.fillna(0, inplace=True)
    df_timeseries['pct_occurrence']   = df_timeseries['pct_occurrence'].astype(int)
    df_timeseries['pct_cloud']        = df_timeseries['pct_cloud'].astype(int)

    # save modified dataframe to its original variable
    self.df_timeseries = df_timeseries[df_columns]

    # garbagge collect
    del df_timeseries
    gc.collect()

    # warning
    print("finished!")


  # merge two or more timeseries
  def merge_timeseries(self, df_list: list):
    df            = pd.concat(df_list, ignore_index=True, sort=False)
    df['index']   = np.arange(start=0, stop=len(df), step=1, dtype=np.int64)
    gc.collect()
    return df.sort_values(by=['year', 'month', 'pixel'])


  # extract image's coordinates and pixels values
  def extract_image_pixels(self, image: ee.Image, month):

    # warning
    if self.seasonal:
      print("Processing season ["+str(month.strftime('%Y'))+"-"+str(self.seasons_list[month.month])+"]...")
    else:
      print("Processing month ["+str(month.strftime('%Y-%m'))+"]...")

    # attributes
    lons_lats_attributes     = None
    cache_files              = self.get_cache_files(month)
    df_timeseries            = pd.DataFrame(columns=self.df_columns)

    # trying to find image in cache
    try:

      # warning - user disabled cache
      if self.force_cache:
        raise Exception()

      # extract pixel values from cache
      lons_lats_attributes  = joblib.load(cache_files[0])
      if lons_lats_attributes is None:
        raise Exception()

      # check if image is empty and return empty dataframe
      if len(lons_lats_attributes) == 0:
        return df_timeseries

    # error finding image in cache
    except:

      # image exists
      try:

        # check if image has less cloud than the cloud threshold
        clip = self.clip_image(self.extract_image_from_collection_monthly(month=month))
        
        # go through each tile
        lons_lats_attributes = np.array([], dtype=np.float64).reshape(0, len(self.attributes)+2)
        for i, geometry in enumerate(self.splitted_geometry):
          print("Extracting geometry ("+str(len(lons_lats_attributes))+") "+str(i+1)+" of "+str(len(self.splitted_geometry))+"...")
          geometry_lons_lats_attributes = gee.extract_latitude_longitude_pixel(image=clip, geometry=geometry, bands=[a+"_water" for a in self.attributes], scale=self.sensor_params['scale'])
          lons_lats_attributes = np.concatenate((lons_lats_attributes, geometry_lons_lats_attributes))

        # save in cache
        joblib.dump(lons_lats_attributes, cache_files[0])

      # error in the extraction process
      except:
        
        # warning
        print("Error while extracting pixels from month "+str(month.strftime('%Y-%m'))+": "+str(traceback.format_exc()))

        # reset attributes
        lons_lats_attributes = None

    # check if has attributes to process
    try:

      # check if they are valid
      if lons_lats_attributes is None:
        raise Exception()

      # build dataframe
      extra_attributes        = np.array(list(zip([0]*len(lons_lats_attributes),[0]*len(lons_lats_attributes),[month.year]*len(lons_lats_attributes),[month.month]*len(lons_lats_attributes),[self.seasons_list[int(month.month)]]*len(lons_lats_attributes))))
      df_timeseries           = pd.DataFrame(data=np.concatenate((extra_attributes, lons_lats_attributes), axis=1), columns=self.df_columns).sort_values(['lat','lon'])
      df_timeseries['pixel']  = range(0,len(df_timeseries))

      # gabagge collect
      del lons_lats_attributes, extra_attributes
      gc.collect()

      # return all pixels in an three pandas format
      return df_timeseries

    # no data do return
    except:

      # show error
      print("Error while extracting pixels month "+str(month.strftime('%Y-%m'))+":")
      print(traceback.format_exc())

      # remove cache file
      if os.path.exists(cache_files[0]):
        os.remove(cache_files[0])
      
      # clear memory
      del lons_lats_attributes
      gc.collect()

      # return empty dataframe
      return df_timeseries


  # save occurrences plot
  def save_occurrences_plot(self, df: pd.DataFrame, folder: str):

    # warning
    print()
    print("Creating occurrences plot to folder '"+folder+"'...")

    # check if folder exists
    if not os.path.exists(folder):
      os.mkdir(folder)

    # number of columns
    columns     = 4
    rows        = 1 if self.seasonal else 3
    fig_height  = 16/columns
  
    # axis ticks
    xticks      = np.linspace(self.sample_lon_lat[0][1], self.sample_lon_lat[1][1], num=4)
    yticks      = np.linspace(self.sample_lon_lat[0][0], self.sample_lon_lat[1][0], num=4)

    # colorbar tixks
    colorbar_ticks_max          = 100
    colorbar_ticks              = np.linspace(0, colorbar_ticks_max if colorbar_ticks_max > 1 else 2, num=5, dtype=int)
    colorbar_ticks_labels       = [str(l) for l in colorbar_ticks]
    colorbar_ticks_labels[-1]   = str(colorbar_ticks_labels[-1])

    # marker size
    multiplier  = math.ceil(self.sensor_params['scale']/100)
    multiplier  = multiplier if multiplier >= 1 else 1
    markersize  = (72./300)*multiplier

    ###############
    ### Monthly Occurrences

    # go through each year
    years_list = list(range(int(self.months_list[0].strftime('%Y')),int(self.months_list[-1].strftime('%Y'))+1))
    for i, year in enumerate(years_list):

      # attributes
      plot_id = 1
      images = []

      # create the plot
      fig = plt.figure(figsize=(20,rows*fig_height), dpi=300)
      fig.suptitle('% Algal Bloom Monthly/Seasonally Occurrences  ('+str(year)+', '+str(gee.indice_selected).upper()+')', fontsize=14, y=1.04)
      fig.autofmt_xdate()
      plt.rc('xtick',labelsize=6)
      plt.rc('ytick',labelsize=6)

      # check seasonal reduction is enabled
      if self.seasonal:

        # go through each season
        for j, season in enumerate(self.seasons_list_):

          # warning
          print("Building occurrences for season '"+str(year)+"/"+str(season)+"'...")

          # filter year data
          df_year = df[(df['year'] == year) & (df['season'] == season)]

          # add plot
          if len(df_year) > 0:
            ax = fig.add_subplot(rows,columns,plot_id)
            ax.grid(True, linestyle='dashed', color='#909090', linewidth=0.1)
            ax.title.set_text(season.capitalize())
            s = ax.scatter(df_year['lat'], df_year['lon'], s=markersize, c=df_year['pct_occurrence'], cmap=plt.get_cmap('jet'))
            s.set_clim(colorbar_ticks[0], colorbar_ticks[-1])
            ax.margins(x=0,y=0)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            images.append(s)
            plot_id = plot_id + 1

      else:

        # go through each month
        for j, month in enumerate(self.months_list):

          # warning
          print("Building occurrences for month '"+str(month.strftime('%Y-%m'))+"'...")

          # check if month is in the year
          if year == month.year:

            # filter year data
            df_year = df[(df['year'] == month.year) & (df['month'] == month.month)]

            # add plot
            if len(df_year) > 0:
              ax = fig.add_subplot(rows,columns,plot_id)
              ax.grid(True, linestyle='dashed', color='#909090', linewidth=0.1)
              ax.title.set_text(month.strftime('%B')+"/"+self.seasons_list[month.month].capitalize())
              s = ax.scatter(df_year['lat'], df_year['lon'], s=markersize, c=df_year['pct_occurrence'], cmap=plt.get_cmap('jet'))
              s.set_clim(colorbar_ticks[0], colorbar_ticks[-1])
              ax.margins(x=0,y=0)
              ax.set_xticks(xticks)
              ax.set_yticks(yticks)
              ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
              ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
              images.append(s)
              plot_id = plot_id + 1

      # figure add cmap
      cbar = fig.colorbar(images[-1], cax=fig.add_axes([0.6, -0.05, 0.39, 0.05]), ticks=colorbar_ticks, orientation='horizontal')
      cbar.set_label("% of occurrence")

      # save it to file
      plt.subplots_adjust(wspace=0.4, hspace=0.4)
      plt.tight_layout()
      fig.savefig(folder+'/occurrences_'+str(year)+'.png', bbox_inches='tight')


    ###############
    ### Monthly Cloud Occurrences

    # go through each year
    years_list = list(range(int(self.months_list[0].strftime('%Y')),int(self.months_list[-1].strftime('%Y'))+1))
    for i, year in enumerate(years_list):

      # attributes
      plot_id = 1
      images = []

      # create the plot
      fig = plt.figure(figsize=(20,rows*fig_height), dpi=300)
      fig.suptitle('% Algal Bloom Monthly/Seasonally Cloud Occurrences  ('+str(year)+')', fontsize=14, y=1.04)
      fig.autofmt_xdate()
      plt.rc('xtick',labelsize=6)
      plt.rc('ytick',labelsize=6)

      # check seasonal reduction is enabled
      if self.seasonal:

        # go through each season
        for j, season in enumerate(['winter','spring','summer','autumn']):

          # warning
          print("Building clouds for season '"+str(year)+"/"+str(season)+"'...")

          # filter year data
          df_year = df[(df['year'] == year) & (df['season'] == season)]

          # add plot
          if len(df_year) > 0:
            ax = fig.add_subplot(rows,columns,plot_id)
            ax.grid(True, linestyle='dashed', color='#909090', linewidth=0.1)
            ax.title.set_text(season.capitalize())
            s = ax.scatter(df_year['lat'], df_year['lon'], s=markersize, c=df_year['pct_cloud'], cmap=plt.get_cmap('Greys'))
            s.set_clim(colorbar_ticks[0], colorbar_ticks[-1])
            ax.margins(x=0,y=0)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            images.append(s)
            plot_id = plot_id + 1

      else:

        # go through each month
        for j, month in enumerate(self.months_list):

          # warning
          print("Building clouds for month '"+str(month.strftime('%Y-%m'))+"'...")

          # check if month is in the year
          if year == month.year:

            # filter year data
            df_year = df[(df['year'] == month.year) & (df['month'] == month.month)]

            # add plot
            if len(df_year) > 0:
              ax = fig.add_subplot(rows,columns,plot_id)
              ax.grid(True, linestyle='dashed', color='#909090', linewidth=0.1)
              ax.title.set_text(month.strftime('%B')+"/"+self.seasons_list[month.month].capitalize())
              s = ax.scatter(df_year['lat'], df_year['lon'], s=markersize, c=df_year['pct_cloud'], cmap=plt.get_cmap('Greys'))
              s.set_clim(colorbar_ticks[0], colorbar_ticks[-1])
              ax.margins(x=0,y=0)
              ax.set_xticks(xticks)
              ax.set_yticks(yticks)
              ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
              ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
              images.append(s)
              plot_id = plot_id + 1

      # figure add cmap
      cbar = fig.colorbar(images[-1], cax=fig.add_axes([0.6, -0.05, 0.39, 0.05]), ticks=colorbar_ticks, orientation='horizontal')
      cbar.set_label("% of occurrence")

      # save it to file
      plt.subplots_adjust(wspace=0.4, hspace=0.4)
      plt.tight_layout()
      fig.savefig(folder+'/occurrences_clouds_'+str(year)+'.png', bbox_inches='tight')

    # warning
    print("finished!")


  # save occurrences geojson
  def save_occurrences_geojson(self, df: pd.DataFrame, path: str):

    # warning
    print()
    print("Saving occurrences geojson to file '"+path+"'...")

    # fix nan values
    df = df.fillna(0)

    # # save occurrences data
    features = []
    for index, row in df.iterrows():
      features.append(geojson.Feature(geometry=geojson.Point((row['lat'], row['lon'])), properties={"year": int(row['year']), "month": int(row['month']), "pct_occurrence": int(row['pct_occurrence']), "pct_cloud": int(row['pct_cloud']), "instants": int(row['instants'])}))
    fc = geojson.FeatureCollection(features)
    f = open(path,"w")
    geojson.dump(fc, f)
    f.close()


  # save a collection in tiff (zip) to folder (time series)
  def save_collection_tiff(self, folder: str, folderName: str, rgb: bool = False):

    # build Google Drive folder name where tiffs will be saved in
    folderName = "abmo_"+str(folderName)+".tiff"
    
    # warning
    print()
    print("Saving image collection in tiff to Folder '"+str(folder)+"' (first try, based on image size) or to your Google Drive at folder '"+str(folderName)+"'...")

    # check if folder exists
    if not os.path.exists(folder):
      os.mkdir(folder)

    # select image attributes to be exported
    attributes = ['occurrence_water', 'not_occurrence_water', 'cloud_water']

    # go through all the collection
    for date in self.dates_timeseries_interval:

      # get image
      image = self.clip_image(self.extract_image_from_collection(date=date), geometry=self.geometry)

      # check if its landsat merge
      if rgb:
        bands = [self.sensor_params['red'], self.sensor_params['green'], self.sensor_params['blue']]+attributes
        if self.sensor_params["sensor"] == "landsat578":
          sensor = image.get(self.sensor_params["property_id"]).getInfo()
          if 'LT05' in sensor:
            bands = [gee.get_sensor_params("landsat5")['red'], gee.get_sensor_params("landsat5")['green'], gee.get_sensor_params("landsat5")['blue']]+attributes
          elif 'LE07' in sensor:
            bands = [gee.get_sensor_params("landsat7")['red'], gee.get_sensor_params("landsat7")['green'], gee.get_sensor_params("landsat7")['blue']]+attributes
          elif 'LC08' in sensor:
            bands = [gee.get_sensor_params("landsat")['red'], gee.get_sensor_params("landsat")['green'], gee.get_sensor_params("landsat")['blue']]+attributes
      else:
        bands = attributes

      # First try, save in local folder
      try:
        print("Trying to save "+date.strftime("%Y-%m-%d")+" GeoTIFF to local folder...")
        image_download_url = image.select(bands).getDownloadUrl({"name": date.strftime("%Y-%m-%d"), "region":self.geometry, "filePerBand": True})
        open(folder+'/'+date.strftime("%Y-%m-%d")+'.zip', 'wb').write(requests.get(image_download_url, allow_redirects=True).content)
        print("finished!")

      # Second try, save in Google Drive
      except:
        print("Error! It was not possible to save GeoTIFF localy. Trying to save it in Google Drive...")
        for band in bands:
          task = ee.batch.Export.image.toDrive(image=image.select(band), folder=folderName, description=date.strftime("%Y-%m-%d")+"_"+str(band), region=self.geometry)
          task.start()
          print(task.status())

    # warning
    print("finished!")
  

  # save a collection in png to folder (time series)
  def save_collection_png(self, folder: str, options: dict = {'min':0, 'max': 3000}):
    
    # warning
    print()
    print("Saving image collection to folder '"+folder+"'...")

    # check if folder exists
    if not os.path.exists(folder):
      os.mkdir(folder)

    # go through all the collection
    for date in self.dates_timeseries_interval:

      # get sensor name
      image_collection = self.collection.filter(ee.Filter.date(date.strftime("%Y-%m-%d"), (date + td(days=1)).strftime("%Y-%m-%d")))

      # check if its landsat merge
      bands = None
      if self.sensor_params["sensor"] == "landsat578":
        sensor = image_collection.first().get(self.sensor_params["property_id"]).getInfo()
        if 'LT05' in sensor:
          bands = [gee.get_sensor_params("landsat5")['red'], gee.get_sensor_params("landsat5")['green'], gee.get_sensor_params("landsat5")['blue']]
        elif 'LE07' in sensor:
          bands = [gee.get_sensor_params("landsat7")['red'], gee.get_sensor_params("landsat7")['green'], gee.get_sensor_params("landsat7")['blue']]
        elif 'LC08' in sensor:
          bands = [gee.get_sensor_params("landsat")['red'], gee.get_sensor_params("landsat")['green'], gee.get_sensor_params("landsat")['blue']]

      # check if folder exists
      path_image = folder+'/'+date.strftime("%Y-%m-%d")
      if not os.path.exists(path_image):
        os.mkdir(path_image)

      # save geometries in folder
      for i, geometry in enumerate(self.splitted_geometry):
        self.save_image(image=self.clip_image(self.extract_image_from_collection(date=date), geometry=geometry), path=path_image+"/"+date.strftime("%Y-%m-%d")+"_"+str(i)+".png", bands=bands, options=options)
    
    # warning
    print("finished!")


  # save a image to file
  def save_image(self, image: ee.Image, path: str, bands: list = None, options: dict = {'min':0, 'max': 3000}):
    
    # warning
    print()
    print("Saving image to file '"+path+"'...")

    # default to RGB bands
    if not bands:
      bands = [self.sensor_params['red'], self.sensor_params['green'], self.sensor_params['blue']]

    # extract imagem from GEE using getThumbUrl function and saving it
    imageIO = PIL.Image.open(BytesIO(requests.get(image.select(bands).getThumbUrl(options), timeout=60).content))
    imageIO.save(path)
    
    # warning
    print("finished!")


  # save a dataset to file
  def save_dataset(self, df: pd.DataFrame, path: str):
    
    # warning
    print()
    print("Saving dataset to file '"+path+"'...")

    # drop unused columns
    df = df.drop(['label'], axis=1)

    # saving dataset to file
    df.to_csv(r''+path, index=False)
    
    # warning
    print("finished!")