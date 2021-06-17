import geojson
import pandas as pd
import glob
import os

# read the results files in data directory
files = glob.glob("data/results*.csv")

# go through all read files
for file in files:
  df = pd.read_csv(r""+file)
  features = []
  for index, row in df.iterrows():
      features.append(geojson.Feature(geometry=geojson.Point((row['lat'], row['lon'])), properties={"occurrence": int(row['occurrence']), "not_occurrence": int(row['not_occurrence']), "pct_occurrence": int(row['pct_occurrence']), "cloud": int(row['cloud']), "pct_cloud": int(row['pct_cloud']), "year": int(row['year']), "season": str(row['season']), "instants": int(row['instants'])}))
  fc = geojson.FeatureCollection(features)
  f = open(os.path.splitext(file)[0].replace("results", "occurrences")+".json","w")
  geojson.dump(fc, f)
  f.close()