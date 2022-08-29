# Databricks notebook source
# 
# This file contains intermediate debiased modeling work used for the 
#   2022 Machine Learning Workshop (part of the Software Symposium)!
#   https://FORWARD_SITE/mlworkshop2022 
#      OR https://INFO_SITE/cdo/events/internal-events/4354c5db-3d3d-4481-97c4-8ad8f12686f1
#
# You can (and should) change them for your own experiments, but they are uniquely defined
# here for constants that we will use together.


# COMMAND ----------

# MAGIC %md
# MAGIC # Let's Make "NYC Taxi 2025"!
# MAGIC <img src='https://images.pexels.com/photos/5648421/pexels-photo-5648421.jpeg?auto=compress&cs=tinysrgb&w=640&h=427&dpr=2' width='300px' title="no more 'hailing' with preemptive taxis" />
# MAGIC 
# MAGIC ## Review
# MAGIC * In the first notebook (**01_explore_data**) we looked at NYC Taxi data and regional demographic data.
# MAGIC    * We found that NYC T&LC's geospatial groupings ("zones") worked well for location aggregation.
# MAGIC    * We found and grouped demographics by zones and found there are some potential biases to deal with
# MAGIC * As a reminder, your code will default to the _'beginner'_ path.  As a testament to your _'expertness'_ you can 
# MAGIC   edit the configuration file (`utilities/settings.py`) to set this variable.  *(HINT: Look for EXPERIENCED_MODE)*
# MAGIC * [Image source](https://www.pexels.com/photo/unrecognizable-black-man-catching-taxi-on-city-road-5648421/)

# COMMAND ----------

# MAGIC %run ../features/featurestore_ops

# COMMAND ----------

# MAGIC %md
# MAGIC ## An Early Classifier
# MAGIC The first part of our worked focused on core CDS responsibilities.   This notebook will look at ways to 
# MAGIC change your raw data and learned ML classifiers.  
# MAGIC 
# MAGIC *NOTE:* For a quantitative review of our progress, we will be training a light ML model, but the principles
# MAGIC for bias mitigation still apply evenif you're just workingn from an XLS spreadsheet!
# MAGIC 
# MAGIC Let's create some features!  From the raw ride data, we know that we have location, time, 
# MAGIC and cost information.  We've alreaady handled location information, so let's think about how to
# MAGIC represent time and cost metrics.
# MAGIC * Day part - instead of specific minutes or hours, let's look at more general day parts
# MAGIC * Weekend or weekday - behaviors may change according to these parts of the week
# MAGIC * Distance travelled - compute this from the specific lat/long coordinates
# MAGIC * Expense of the fare - there's still that data problem (check out the previous notebook) that we need to solve for
# MAGIC * (location zone) - also include the mapped "zone" for the features now that we're chosen them

# COMMAND ----------

# load geometry for zip codes and filter for NEW YORK state; 
df_shape_tlc = spark.read.format('delta').load(CREDENTIALS['paths']['geometry_nyctaxi'])


# only admins write this one (it takes almost 10m to aggregate)
if CREDENTIALS['constants']['EXPERIENCED_MODE'] and CREDENTIALS['constants']['WORKSHOP_ADMIN_MODE']:  
    ### --- these are parts from the first notebook ---- 
    # encode the cells (to match zones)
    df_tlc_cells = shape_encode_h3cells(df_shape_tlc, ['zone'], CREDENTIALS['constants']['RESOLUTION_H3'], 'the_geom')
    # load taxi data that has h3 coordinates
    df_taxi_encoded = spark.read.format('delta').load(CREDENTIALS['paths']['nyctaxi_h3_sampled'])  # has h3

    # join for both the pick-up and drop off
    df_taxi_indexed = (
        point_intersect_h3cells(df_taxi_encoded.withColumnRenamed('pickup_h3', 'h3'),   # temp rename
                                'pickup_latitude', 'pickup_longitude', 
                                CREDENTIALS['constants']['RESOLUTION_H3'], df_tlc_cells, col_h3='h3')
        .withColumnRenamed('zone', 'pickup_zone')   # rename merged zip
        .withColumnRenamed('h3', 'pickup_h3')   # fix rename and drop extra
    )
    df_taxi_indexed = (
        point_intersect_h3cells(df_taxi_indexed.withColumnRenamed('dropoff_h3', 'h3'),   # temp rename
                                'dropoff_latitude', 'dropoff_longitude', 
                                CREDENTIALS['constants']['RESOLUTION_H3'], df_tlc_cells, col_h3='h3')
        .withColumnRenamed('zone', 'dropoff_zone')   # rename merged zip
        .withColumnRenamed('h3', 'dropoff_h3')    # fix rename and drop extra
    )

    ### --- now we're adding new features ---- 
    df_taxi_indexed = (df_taxi_indexed
        # encode trip duration 
        .withColumn('ride_duration', (F.col("dropoff_datetime") - F.col("pickup_datetime")).seconds)
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
        .withColumn('day_of_week', F.dayofweek(F.col('pickup_datetime')))
        .withColumn('weekpart', 
                    F.when((F.col('day_of_week')==F.lit(1)) & (F.col('day_of_week')==F.lit(7)), 'weekend')  # Sun, Sat
                    .when((F.col('day_of_week')==F.lit(6)) & (F.col('pickup_daypart')==F.lit('night')), 'weekend')  # Friday + night
                    .otherwise('weekday'))  # all other times
        # encode into longer term, like month and week of year
        .withColumn('month_of_year', F.month(F.col('pickup_datetime')))
        .withColumn('week_of_year', F.weekofyear(F.col('pickup_datetime')))          
    )

# COMMAND ----------

path_read = CREDENTIALS['paths']['nyctaxi_h3_zones']

# only admins write this one (it takes almost 10m to aggregate)
if CREDENTIALS['constants']['EXPERIENCED_MODE'] and CREDENTIALS['constants']['WORKSHOP_ADMIN_MODE']:  
    # now filter to relevant zips (those found in our taxi data)
    df_zones = (df_taxi_indexed
        .select(F.col('pickup_zone').alias('zone'))   # first sum by pickup
        .groupBy('zone').agg(F.count('zone').alias('count'))
        .union(df_taxi_indexed.select(F.col('dropoff_zone').alias('zone'))   # and then by dropoff
            .groupBy('zone').agg(F.count('zone').alias('count'))
        )
        .groupBy('zone').agg(F.sum('count').alias('count'))  # sum them all together
        .withColumnRenamed('zone', '_zone_join')
    )
    df_zones = (df_zones
        .join(df_shape_tlc, df_zones['_zone_join']==df_shape_tlc['zone'], 'inner')
        .drop('_zone_join')
        .withColumn('count_log10', F.log(10.0, F.col('count')))
    )
    dbutils.fs.rm(path_read, True)
    df_zones.write.format('delta').save(path_read)

fn_log("Grouping into a plottable dataframe...")
df_zones = spark.read.format('delta').load(path_read)
pdf_sub = df_zones.toPandas().sort_values(by='count', ascending=False)
pdf_sub['geometry'] = pdf_sub['geometry'].apply(lambda x: wkt.loads(x))
num_total = len(pdf_sub['count'])
shape_plot_map(pdf_sub, col_viz='count_log10', txt_title=f"Zone Log-Count ({num_total} total zones)", 
               gdf_background=pdf_shape_states)


# COMMAND ----------

# MAGIC %md
# MAGIC # Want to find how to make a predictor
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
