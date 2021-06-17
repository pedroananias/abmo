import geojson
import pandas as pd
import glob



# df = pd.read_csv(r"data/results[moc=4].csv")
# features = []
# for index, row in df.iterrows():
#     features.append(geojson.Feature(geometry=geojson.Point((row['lat'], row['lon'])), properties={"occurrence": int(row['occurrence']), "not_occurrence": int(row['not_occurrence']), "pct_occurrence": int(row['pct_occurrence']), "cloud": int(row['cloud']), "pct_cloud": int(row['pct_cloud']), "year": int(row['year']), "season": str(row['season']), "instants": int(row['instants'])}))
# fc = geojson.FeatureCollection(features)
# f = open("data/occurrences[moc=4].json","w")
# geojson.dump(fc, f)
# f.close()