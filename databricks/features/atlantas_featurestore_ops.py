# Databricks notebook source
# MAGIC %run ../utilities/settings

# COMMAND ----------

import geopandas as gpd
import pyspark.sql.functions as F
import pandas as pd


# COMMAND ----------

# load demographics from neustar/feature store, dropping personally identifiable field (e.g. name, custloc, etc)
def demographic_load_fs(list_columns=['zipcd', 'age', 'hshld_income_cd', 'est_curr_home_value', 'exact_age', 'gnrt', 'lang', 'ethnc_grp', 'geohash_lat_lng', 'building_area_1_cif', 'lot_size_square_feet_cif', 'market_value_land_cif', 'market_value_improvement_cif', 'number_of_units_cif', 'number_of_bedrooms_cif', 'year_built_cif', 'tax_amount_cif', 'length_of_residence_cif', 'state_cif', 'city_cif']):
    from featurestore import *
    import os
    os.environ['no_proxy'] = 'atlantis-feature-store.SERVICE_SITE'
    client = Client('atlantis-feature-store.SERVICE_SITE:8080')

    # Token='Enter your PAT token' --- see the configure file
    client.auth.set_auth_token(CREDENTIALS['credentials']['ATLANTIS'])

    project = client.projects.get('Neustar')
    fs = project.feature_sets.get('Neustar Demographics')
    data_reference = fs.retrieve()
    df = data_reference.as_spark_frame(spark)

    # filter to specifically requested columns
    if list_columns is not None:
        return df.select(list_columns)
    return df

# testing functions to load/save demographics
if False: 
    path_write = "abfss://mlworkshop2022@dsairgeneraleastus2sa.STORAGE/demographics"
    df_sub = demographic_load_fs()
    df_sub.write.format('delta').save(path_write)
    df_sub = spark.read.format('delta').load(path_write)

# COMMAND ----------

path_read = "abfss://mlworkshop2022@dsairgeneraleastus2sa.STORAGE/nyctaxi/tables/nyctaxi_yellow"
df = spark.read.format('delta').load(path_read)
print(df.count())
display(df)


# COMMAND ----------

import geopandas as gpd
from h3ronpy import vector, util
import matplotlib.pyplot as plt
import math

#helper to demonstrate the different resolutions
def shp_plot_example(gpd_example=None, res_compare=[7, 9, 11]):
    # containment example - https://observablehq.com/@nrabinowitz/h3-hierarchical-non-containment?collection=@nrabinowitz/h3
    if gpd_example is None:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        gpd_example = world[world["continent"] == "Africa"]
        res_compare = [2, 3, 4]
    
    # original source - https://towardsdatascience.com/uber-h3-for-data-analysis-with-python-1e54acdcc908
    figure, axis = plt.subplots(2, 2, figsize=(9,9))
    gpd_copy = gpd.GeoDataFrame(gpd_example)
    gpd_copy.plot(ax=axis[0, 0])
    axis[0,0].set_title('original')
    print(gpd_example.columns)

    # convert to h3 cells and build a new geodataframe
    for res_idx in range(1,4):
        ax_target = axis[res_idx % 2, math.floor(res_idx / 2)]
        gdf = util.h3index_column_to_geodataframe(vector.geodataframe_to_h3(gpd_example, res_compare[res_idx-1]))
        del gpd_example['to_h3_idx']
        gdf.plot(ax=ax_target)
        ax_target.set_title(f"resolution: {res_compare[res_idx-1]}")

    
# https://github.com/nmandery/h3ronpy
# consider other shape sources

# plotting different colors, like heatmap -- https://www.kaggle.com/code/imdevskp/geopandas
shp_plot_example()

# COMMAND ----------

import fsspec
import pandas
import geopandas as gpd

# method to read shp directly from ADLS (unfortunately, it didn't work in the end)
def shp_import_adls(path_file='abfss://mlworkshop2022@dsairgeneraleastus2sa.STORAGE/tl_2021_us_zcta520/tl_2021_us_zcta520/tl_2021_us_zcta520.shp', 
                    mode='rb', sas_token=None):
    import re
    # Example shape data location
    # tl_2021_us_zcta520.shp.ea.iso.xml "https://dsairgeneraleastus2sa.blob.core.windows.net/mlworkshop2022/shapefiles

    # extra_configs = {
    #   "fs.azure.account.auth.type": "CustomAccessToken",
    #   "fs.azure.account.custom.token.provider.class": spark.conf.get("spark.databricks.passthrough.adls.gen2.tokenProviderClassName"),
    #   "token_credential": spark.conf.get("spark.databricks.passthrough.adls.gen2.tokenProviderClassName")
    # }
    if sas_token is None:
        sas_token = "sp=rl&st=2022-08-21T21:09:53Z&se=2022-08-23T05:09:53Z&spr=https&sv=2021-06-08&sr=d&sig=TVium1rK%2B6I89hSk6kj%2Fv0%2BMkS14YIFCA9g0OA2iq4E%3D&sdd=1"

    adls_account_name = 'dsairgeneraleastus2sa'

    adls_path = f'abfss://mlworkshop2022@dsairgeneraleastus2sa.STORAGE/tl_2021_us_zcta520/{path_file}'
    fsspec_handle = fsspec.open(adls_path, account_name=adls_account_name, anon=False, account_key=adls_account_key, mode=mode) # account_name = adls_account_name, sas_token=sas_key)

#     with fsspec_handle.open() as f:
#         textf = f.read()
#     print(textf)

    # tl_2021_read(path_file='tl_2021_us_zcta520.shp.iso.xml').read()
    return fsspec_handle

# print(tl_2021_read('tl_2021_us_zcta520.shp.iso.xml', 'rt').open().read())

# https://towardsdatascience.com/uber-h3-for-data-analysis-with-python-1e54acdcc908
# warning, this will read a large amount of memory into place...
# with tl_2021_read(mode='rb').open() as file:
#     united_states = gpd.read_file(file)

# COMMAND ----------

import geopandas as gpd
from h3ronpy import vector, util

# https://towardsdatascience.com/uber-h3-for-data-analysis-with-python-1e54acdcc908
# warning, this will read a large amount of memory into place...
with tl_2021_read(mode='rb').open() as file:
    united_states = gpd.read_file(file)
# united_states = gpd.read_file(tl_2021_read(mode='rb').open())
# africa = world[world["continent"] == "Africa"]
# africa.plot(column="name")

# https://github.com/nmandery/h3ronpy
# consider other shape sources

# convert to h3 cells and build a new geodataframe
# gdf = util.h3index_column_to_geodataframe(vector.geodataframe_to_h3(africa, 3))
# gdf.plot(column="name")

# plotting different colors, like heatmap -- https://www.kaggle.com/code/imdevskp/geopandas

# COMMAND ----------


def shape_export(path_shp, path_csv):
    gdf_new = gpd.read_file(path_shp)
    df_new = pd.DataFrame(gdf_new)
    df_new.to_csv(path_csv)

# COMMAND ----------

def zip_import(path_csv='abfss://mlworkshop2022@dsairgeneraleastus2sa.STORAGE/tl_2021_us_zcta520/raw/tl_2021_us_zcta520.csv',
               path_df_zip='abfss://mlworkshop2022@dsairgeneraleastus2sa.STORAGE/tl_2021_us_zcta520/geometry/ztca',
               col_zip="ZCTA5CE20"):
    import zipcodes
    import re
    import json

    fn_log(f"[zip_import] Reading CSV from '{path_csv}'....")
    df_with_schema = (
        spark.read.format("csv")
        .option("inferSchema" , True)
        .option("header", True)
        .load(path_csv)
        .drop('_c0')
        .withColumn('geometry', F.col('geometry').cast('string'))
        .withColumnRenamed(col_zip, 'zip')
        # .withColumn('geo_processed', convertUDF(F.col('geometry')))  --- wait until final component
    )
    
    # rename from MTFCC20 -> mtfcc
    re_sub = re.compile(r"[\d]+$")
    for x in df_with_schema.columns:
        if re_sub.search(x) is not None:
            df_with_schema = df_with_schema.withColumnRenamed(x, re_sub.sub('', x.lower()))
    
    # now join for zip code information...
    def zip_lookup(z):
        if not zipcodes.is_real(z):
            return None
        exact_zip = zipcodes.matching(z)
        if len(exact_zip):
            exact_zip = exact_zip[0]
            return json.dumps(exact_zip)
        return None

    # keep columns - https://pypi.org/project/zipcodes/
    fn_log(f"[zip_import] Binding to zip locations...")
    zip_expand = F.udf(zip_lookup)
    df_final = (df_with_schema
        .withColumn('lookup', F.from_json(zip_expand(F.col('zip').cast('string')), "MAP<STRING,STRING>"))
        # list_json_cols = ['city', 'country', 'state', 'timezone', 'zip_code_type', 'area_codes']
        .withColumn('city', F.col('lookup.city'))
        .withColumn('country', F.col('lookup.country'))
        .withColumn('timezone', F.col('lookup.timezone'))
        .withColumn('state', F.col('lookup.state'))
        .withColumn('zip_code_type', F.col('lookup.zip_code_type'))
        .withColumn('area_codes', F.col('lookup.area_codes'))
        .withColumn('acceptable_cities', F.col('lookup.acceptable_cities'))
        .withColumn('unacceptable_cities', F.col('lookup.unacceptable_cities'))
        .drop('lookup')
    )
    # write out to final destination
    fn_log(f"[zip_import] Write to final delta location '{path_df_zip}'...")
    df_final.write.format('delta').mode('overwrite').option('overwriteSchema', 'true').save(path_df_zip)
    df_final = spark.read.format('delta').load(path_df_zip)
    return df_final

# df_zip_shape = zip_import()
# display(df_zip_shape)

# COMMAND ----------

# attempt to map zip code to a source DF using lat/lon and h3 indexing
def h3_map_zip(df_src, df_zip, col_src_lat, col_src_lon, col_zip_lat='intptlat', col_zip_lon='intptlon', h3_resolution=11):
    # confused about resolution? check out this example - https://observablehq.com/@nrabinowitz/h3-hierarchical-non-containment?collection=@nrabinowitz/h3
    
    
    .withColumn('h3', udf_h3(F.col('intptlat'), F.col('intptlon'), F.lit(8)))
    
    

# COMMAND ----------

from h3 import h3 # import vector, util
# https://h3geo.org/docs/api/indexing#geotoh3
# h3.geo_to_h3(lat, lng, resolution)
udf_h3 = F.udf(h3.geo_to_h3)

df_sub = (df_zip_shape
    .filter(F.col('state') == F.lit("TX"))
    .withColumn('h3', udf_h3(F.col('intptlat'), F.col('intptlon'), F.lit(8)))
)
display(df_sub)



# COMMAND ----------

import geopandas as gpd
from shapely import wkt
df_zip_shape = spark.read.format('delta').load('abfss://mlworkshop2022@dsairgeneraleastus2sa.STORAGE/tl_2021_us_zcta520/geometry/ztca')
pdf_sub = (df_zip_shape
    .filter(F.col('zip') == F.lit(78759))
    .toPandas()
)
display(pdf_sub)
pdf_sub['geometry'] = pdf_sub['geometry'].apply(lambda x: wkt.loads(x))
print(pdf_sub['geometry'])
shp_plot_example(pdf_sub, res_compare=[7, 9, 11])
# gdf2_us = gpd.GeoDataFrame(pdf_sub)
# gdf2_us.plot()


from h3ronpy import vector, util
# convert to h3 cells and build a new geodataframe
# gdf = util.h3index_column_to_geodataframe(vector.geodataframe_to_h3(gdf2_us, 8))
# gdf.plot()


# COMMAND ----------


