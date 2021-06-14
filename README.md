# Algal Bloom Monthly Occurrence

Module responsable for extracting Algal Bloom Monthly Occurences in a region of interest based on images from Google Earth Engine API



### Dependencies

- Python 3.7.7 64-bit ou superior
- Modules: oauth2client earthengine-api matplotlib pandas numpy requests pillow natsort geojson argparse logging hashlib joblib gctraceback warnings



### Attention, before running:

Before running the script and after installing the libraries, you must authenticate with the Google Earth Engine API using the following command:

```
earthengine authenticate
```



### How to execute the default script?

python /path/to/abmo/script.py --lat_lon=-48.84725671390528,-22.04547298853004,-47.71712046185493,-23.21347463046867 --name=bbhr --date_start=1985-01-01 --date_end=2018-12-31




### What are the results?

The script will generate annual results of occurrence of algae and clouds blooming in the inserted study area. Therefore, a folder located in 'data' is created and named based on the date and version of the script executed. Example: /path/to/abmo/data/20200701_111426[v=V1-bbhr,date_start=1985-01-01,date_end=2001-12-31]. 

ATTENTION: big date range tends to lead to memory leak, stopping the script execution. It is always a good pratice to split the dates in two or three parts, unless you have a big amount of memory in your computer.

The following results are generated:

- timeseries.csv (Monthly time series of pixels, year, latitude, longitude and occurrences of algae bloom (based on the threshold of the Slope index) and clouds)
- occurrences.png (Graphs of occurrences separated annually from algal blooms)
- occurrences_clouds.png (Graphs of occurrences separated annually from clouds)
- occurrences.json (GeoJSON of occurrences by year and month with parameters (occurrence, not_occurrence, pct_occurrence, cloud, pct_cloud, year, month and instants) that can be imported into QGIS and filtered)
- tiff/{date}.zip (ZIP of GeoTIFFs comprehending bands Cloud/Only Water Body, Occurrence/Only Water Body and Not Occurrence/Only Water Body of the study area. If the script can not save those images in locally, will send them to Google Drive)


### Exporting GeoTIFFs to Google Drive

When using the 'save_collection_tiff' function, if the script can not save images locally, will send them to Google Drive to a folder called 'abmo_name_version.tiff' for user who is authenticated. Daily images used in the composition of the monthly time series are saved. These images will be separated by the following bands: Red, Green, Blue, Cloud/Only Water Body, Occurrence/Only Water Body and Not Occurrence/Only Water Body. However, after running the Abmo script, images are likely to take a while to be inserted into Drive due to processing time. It is necessary to wait approximately 1 day until they are all available, depending on the size of the study area.



### Example

```
# Import
import ee
from modules import abmo

# Initialize Google Earth Engine
ee.Initialize()

# folder where to save results
folder = "/path/to/desired/folder"

# create algorithm object
abmo = abmo.Abmo(lat_lon="-48.84725671390528,-22.04547298853004,-47.71712046185493,-23.21347463046867",
                date_start=dt.strptime("1985-01-01", "%Y-%m-%d"),
                date_end=dt.strptime("2018-12-31", "%Y-%m-%d"),
                sensor="landsat578",
                cache_path=folder, 
                force_cache=False,
                indice="mndwi,ndvi,fai,sabi,slope",
                min_occurrence=4)

# creating yearly timeseries
abmo.process_timeseries_data()

# save timeseries in csv file
abmo.save_dataset(df=abmo.df_timeseries, path=folder+'/timeseries.csv')

# save ocurrences plot (monthly and clouds)
abmo.save_occurrences_plot(df=abmo.df_timeseries, folder=folder)

# save geojson occurrences
abmo.save_occurrences_geojson(df=abmo.df_timeseries, folder+"/geojson")

# save daily images to Google Drive
abmo.save_collection_tiff(folder=folder+"/tiff", folderName="bbhr")
```