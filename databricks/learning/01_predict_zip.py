# Databricks notebook source
# MAGIC %run ../features/location_ops

# COMMAND ----------

import geopandas as gpd
import pyspark.sql.functions as F
import pandas as pd


# COMMAND ----------


# load our sample data (NYC taxi paths)
path_read = CREDENTIALS['paths']['nyctaxi_raw']
df_taxi_raw = spark.read.format('delta').load(path_read)
fn_log(f"Taxi data columns: {df_taxi_raw.columns}")
# display(df_taxi_raw)

# load geometry within new york state
path_read = CREDENTIALS['paths']['geometry_zip']
df_geo = spark.read.format('delta').load(path_read).filter(F.col('state')==F.lit('NY'))

# decode geometry's zip files so we can easily perform lat/long intersections (note, must be in Pandas first)
from shapely import wkt, geometry
import json

pdf_geo_decode = lambda x: json.dumps(geometry.mapping(wkt.loads(x)))
udf_geo_decode = F.udf(pdf_geo_decode)
df_geo = df_geo.withColumn('geometry_json', udf_geo_decode(F.col('geometry')))

# COMMAND ----------

# MAGIC %md
# MAGIC # Map source data to geo coordinates

# COMMAND ----------

import h3ronpy

# now perform mapping down to some h3 cells (resolution 11?)
#    according to h3 resolution - https://h3geo.org/docs/core-library/restable/#average-area-in-m2
#    resolution 11 is about 23138.565 square feet
#    resolution 10 (roughly a city block) is about 161970 square feet

resolution_h3 = 11

# https://pypi.org/project/h3-pyspark/
from h3_pyspark.indexing import index_shape 

from shapely import wkt, geometry
import json

# create encoding function from raw zip code geometry to h3
pdf_geo_decode = lambda x: json.dumps(geometry.mapping(wkt.loads(x)))
udf_geo_decode = F.udf(pdf_geo_decode)

df_encoded = (spark.read.format('delta').load(CREDENTIALS['paths']['geometry_zip'])
    .filter(F.col('state')==F.lit('NY'))
    .withColumn('geometry_json', udf_geo_decode(F.col('geometry')))
    .withColumn(f"h3", index_shape(F.col('geometry_json'), F.lit(resolution_h3)))
    .select('zip', 'geoid', F.explode('h3').alias('h3'))
)
# display(df_encoded.limit(10))

# now encode taxi trips from raw lat/long to h3
df_taxi_indexed = (df_taxi_raw
    .withColumn('h3_pickup', h3_pyspark.geo_to_h3(F.col('pickup_latitude'), F.col('pickup_longitude'), F.lit(resolution_h3)))
    .withColumn('h3_dropoff', h3_pyspark.geo_to_h3(F.col('dropoff_latitude'), F.col('dropoff_longitude'), F.lit(resolution_h3)))
)

# finally, join the zip geometry to the lat/long
df_taxi_indexed = (df_taxi_indexed
    .join(df_encoded.withColumnRenamed('zip', 'zip_pickup'), 
          df_encoded['h3']==df_taxi_indexed['h3_pickup']).drop('geoid', 'h3')
    .join(df_encoded.withColumnRenamed('zip', 'zip_dropoff'), 
          df_encoded['h3']==df_taxi_indexed['h3_dropoff']).drop('geoid', 'h3')
)

# write indexed geo taxi
fn_log(f"Writing geo-encoded taxi data...")
df_taxi_raw = df_taxi_indexed.write.format('delta').save(CREDENTIALS['paths']['nyctaxi_geo'])

# from h3ronpy import vector, util
# gdf_plot = util.h3index_column_to_geodataframe(vector.geodataframe_to_h3(gdf_copy, res_compare[res_idx]))

                           

# pdf_geo['geometry'] = pdf_geo['geometry'].apply(lambda x: wkt.loads(x))
# fn_log(f"Geometry data columns: {pdf_geo.columns}")
# display(df_geo)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Want to find how to make a predictor
# MAGIC ## Map to zip code
# MAGIC doesn't work, too broad
# MAGIC 
# MAGIC ## Revisit other data, map to NYC taxi
# MAGIC looks good
# MAGIC 
# MAGIC ## Wrap-up
# MAGIC * Save data for prediction with these constraints
# MAGIC * need to explore a little for your best join
# MAGIC 
# MAGIC # Want to do learning to predict
# MAGIC 
# MAGIC ## grouping by the right criterion
# MAGIC * grouping to the right time
# MAGIC 
# MAGIC ## first evaluation
# MAGIC * waht does first model tell me
# MAGIC * good to go, right?
# MAGIC 
# MAGIC ## whoa, looks biased ....
# MAGIC * against borough
# MAGIC * aginst income
# MAGIC * etc...

# COMMAND ----------

fn_log(f"Writing geo-encoded taxi data...")
df_taxi_indexed.write.format('delta').save(CREDENTIALS['paths']['nyctaxi_geo'])
df_taxi_indexed = spark.read.format('delta').load(CREDENTIALS['paths']['nyctaxi_geo'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check to see unique zip codes
# MAGIC Uh oh, this includes all of NY state and that's not what we want

# COMMAND ----------

# now filter to relevant zips (those found in our taxi data)
df_zips = (df_taxi_indexed
    .select(F.col('zip_pickup').alias('zip'))
    .union(df_taxi_indexed.select(F.col('zip_dropoff').alias('zip')))
    .groupBy('zip').agg(F.count('zip').alias('count'))
    .orderBy(F.col('count').desc())
)
fn_log(f"After pooling, we found {df_zips.count()} unique zips...")
display(df_zips)

from shapely import wkt

fn_log(f"Plotting of unique zips on map...")

df_geo = spark.read.format('delta').load(CREDENTIALS['paths']['geometry_zip'])
df_geo = (df_geo
    .join(df_zips, df_zips['zip']==df_geo['zip'], 'inner')
    .withColumn('count_log10', F.log(10.0, F.col('count')))
)
pdf_sub = df_geo.toPandas()
pdf_sub['geometry'] = pdf_sub['geometry'].apply(lambda x: wkt.loads(x))
shape_plot_map(pdf_sub, col_viz='count_log10')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Now, join other data from taxi shape
# MAGIC This will limit those zip codes that map to NYC taxi service, not the greater state...

# COMMAND ----------

df_geo_nyctax = spark.read.format('delta').load(CREDENTIALS['paths']['geometry_nyctaxi'])
df_geo_filtered = (df_geo
    .join(df_zips, df_zips['zip']==df_geo['zip'], 'inner')
    .withColumn('count_log10', F.log(10, F.col('count')))
)


# COMMAND ----------

df_taxi_zip = (df_taxi_indexed
    # encode into hour parts
    .withColumn('pickup_hour', F.hour(F.col('pickup_datetime')))
    .withColumn('pickup_daypart', 
                F.when(F.col('pickup_hour') < F.lit(5), 'twilight')  # 12a-5a
                .when(F.col('pickup_hour') < F.lit(10), 'morning')  # 5a-10a
                .when(F.col('pickup_hour') < F.lit(15), 'lunch')  # 10a-3p
                .when(F.col('pickup_hour') < F.lit(18), 'afternoon')  # 3p-6p
                .when(F.col('pickup_hour') < F.lit(18), 'night')  # 6p-11p
                .otherwise('twilight'))  # 11p-12a
    .withColumn('dropoff_hour', F.hour(F.col('dropoff_datetime')))
    .withColumn('dropoff_daypart', 
                F.when(F.col('dropoff_hour') < F.lit(5), 'twilight')  # 12a-5a
                .when(F.col('dropoff_hour') < F.lit(10), 'morning')  # 5a-10a
                .when(F.col('dropoff_hour') < F.lit(15), 'lunch')  # 10a-3p
                .when(F.col('dropoff_hour') < F.lit(18), 'afternoon')  # 3p-6p
                .when(F.col('dropoff_hour') < F.lit(18), 'night')  # 6p-11p
                .otherwise('twilight'))  # 11p-12a
    # encode into day of week
    .withColumn('pickup_dow', F.dayofweek(F.col('pickup_datetime')))
    .withColumn('pickup_weekpart', 
                F.when((F.col('pickup_dow')==F.lit(1)) & (F.col('pickup_dow')==F.lit(7), 'weekend')  # Sun, Sat
                .when((F.col('pickup_dow')==F.lit(6)) & (F.col('pickup_daypart')==F.lit('night'), 'weekend')  # Friday + night
                .otherwise('weekday'))  # all other times
    # encode into day of week
    .withColumn('dropoff_dow', F.dayofweek(F.col('dropoff_datetime')))
    .withColumn('dropoff_weekpart', 
                F.when((F.col('dropoff_dow')==F.lit(1)) & (F.col('dropoff_dow')==F.lit(7), 'weekend')  # Sun, Sat
                .when((F.col('dropoff_dow')==F.lit(6)) & (F.col('dropoff_daypart')==F.lit('night'), 'weekend')  # Friday + night
                .otherwise('weekday'))  # all other times
              
)

# COMMAND ----------

path_read = CREDENTIALS['paths']['demographics_raw']
df_demographics = spark.read.format('delta').load(path_read)
fn_log(f"Total available demographic entries... {df_demographics.count()}")
display(df)

# find zips of interest
df_zips = df_demographics.select('zipcd').distinct()
fn_log(f"Number of unique zips... {df_zips.count()}")


# COMMAND ----------

# load raw geo
# encode   latitude_cif', 'longitude_cif , resolution 11
# load cab rides
# encode pickup and drop off at 11
# encode pickup time into morning / afternoon / evening
# encode pickup time by day of week
# partition before/after for train/test
# predict the locations that will be most profitable morning / afternoon / evening on days of week
# generate model that predicts by areas

# overlay biggest demographic classes by age or race or income
# compare disparities for evening pickups

# solution 1 - create model to better sample

# solution 2- postfix to hit min score
