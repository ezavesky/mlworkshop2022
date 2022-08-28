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
# MAGIC # Let's Make "NYC Taxi 2035"!
# MAGIC <img src='https://images.pexels.com/photos/5648421/pexels-photo-5648421.jpeg?auto=compress&cs=tinysrgb&w=640&h=427&dpr=2' width='300px' title="no more 'hailing' with preemptive taxis" />
# MAGIC 
# MAGIC ## Background
# MAGIC * In the future, there will be a need to preemptively position resources, like autonomous 
# MAGIC   cars, drones, groceries, etc. Where they may be most needed and right when they're needed
# MAGIC * All of this preemptiveness is accomplished with AI and machine-learned models built with historical data.
# MAGIC 
# MAGIC This is your chance to build these insights together!! 
# MAGIC 
# MAGIC ### Ground Rules
# MAGIC * In this workshop, we'll start with your CDS data (e.g. taxi rides) and discover ways to augment
# MAGIC   with location, demographic, and other datasets to make better predictions.
# MAGIC * In these notebooks, there will be both a 'beginner' and an 'experienced' set of code that you can 
# MAGIC   execute according to your preference.  
# MAGIC     * The big difference is that the 'beginner' code will mostly load pre-computed data and focus 
# MAGIC       on the illustration of that data instead of the algorithms itself.
# MAGIC * By default, your code will run the _'beginner'_ path.  As a testament to your _'expertness'_ you can 
# MAGIC   edit the configuration file (`utilities/settings.py`) to set this variable.  *(HINT: Look for EXPERIENCED_MODE)*
# MAGIC * [Image source](https://www.pexels.com/photo/unrecognizable-black-man-catching-taxi-on-city-road-5648421/)

# COMMAND ----------

# MAGIC %run ../features/featurestore_ops

# COMMAND ----------

## Just for Visualization ##
# We won't walk through this code, but it gives a peak at how we compute some aggregate
# statistics for the dataset.  We avoid the execution here because it takes several minutes
# for each person to re-run the statistics on this large dataset

def dataset_filter_raw(df_input):
    import datetime as dt
    # this is a helper function to filter out bad data by time range
    return (df_input
        .filter(F.col('pickup_datetime') >= F.lit(dt.datetime(year=2009, day=1, month=1)))  # after Jan 1, 2009
        .filter(F.col('pickup_datetime') <= F.lit(dt.datetime(year=2020, day=1, month=1)))  # before Jan 1, 2020
        .dropna(subset=['fare_amount'])  # drop any records that have empty/null fares
        .filter(F.col('fare_amount') >= F.lit(0.0))  # no negative-dollar fares
        .filter(F.col('fare_amount') <= F.lit(20000.0))  # no super-expensive fares
    )

# an example of aggregations requried to build a simple time plot
path_read = CREDENTIALS['paths']['nyctaxi_stats']
if CREDENTIALS['constants']['EXPERIENCED_MODE'] and CREDENTIALS['constants']['WORKSHOP_ADMIN_MODE']:   
    # preprocessing if admin (it's a lot of data)
    path_read_raw = CREDENTIALS['paths']['nyctaxi_raw']
    df_taxi_stats = (
        dataset_filter_raw(spark.read.format('delta').load(path_read_raw))
        .withColumn('date_trunc', F.date_trunc('day', F.col('pickup_datetime')))
        .groupBy('date_trunc').agg(
            F.mean(F.col('fare_amount')).alias('mean_total'),
            F.max(F.col('fare_amount')).alias('max_total'),
            F.min(F.col('fare_amount')).alias('min_total'),
            F.count(F.col('pickup_datetime')).alias('volume'),        
        )
    )
    dbutils.fs.rm(path_read, True)  # destory entirely
    df_taxi_stats.write.format('delta').save(path_read)

# load data, make sure it's sorted by date!
pdf_taxi_stats = (spark.read.format('delta').load(path_read)
    .toPandas()
    .sort_values('date_trunc')
)

# graph example from great plotting how-to site...
#    https://www.python-graph-gallery.com/line-chart-dual-y-axis-with-matplotlib

# https://matplotlib.org/stable/gallery/color/named_colors.html
COLOR_VOLUME = 'tab:blue'
COLOR_PRICE = 'tab:orange'

fig, ax1 = plt.subplots(figsize=(14, 8))
ax2 = ax1.twinx()
ax1.bar(pdf_taxi_stats['date_trunc'], pdf_taxi_stats['volume'], color=COLOR_VOLUME, width=1.0)
ax2.plot(pdf_taxi_stats['date_trunc'], pdf_taxi_stats['mean_total'], color=COLOR_PRICE, lw=4)
# want to see some other weird stats? min/max are crazy (but we'll fix that later)
# ax2.plot(pdf_taxi_stats['date_trunc'], pdf_taxi_stats['min_total'], color='violet', lw=2)
# ax2.plot(pdf_taxi_stats['date_trunc'], pdf_taxi_stats['max_total'], color='firebrick', lw=2)

ax1.set_xlabel("Date")
ax1.set_ylabel("Total Ride Volume", color=COLOR_VOLUME, fontsize=14)
ax1.tick_params(axis="y", labelcolor=COLOR_VOLUME)

ax2.set_ylabel("Average Total Fare ($)", color=COLOR_PRICE, fontsize=14)
ax2.tick_params(axis="y", labelcolor=COLOR_PRICE)

fig.autofmt_xdate()
fig.suptitle("2009 NYC Taxi Volume and Average Fares", fontsize=20)


# compute some overall stats
fn_log(f"Date Range: {pdf_taxi_stats['date_trunc'].min()} - {pdf_taxi_stats['date_trunc'].max()} (total days {len(pdf_taxi_stats)})")
fn_log(f"Total Rides: {pdf_taxi_stats['volume'].sum()}")
fn_log(f"Avg Fare: {pdf_taxi_stats['mean_total'].mean()}")



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

df_taxi_zip = (df_taxi_indexed
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
