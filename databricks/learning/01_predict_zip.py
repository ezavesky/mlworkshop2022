# Databricks notebook source
# 
# This file contains introductory modeling work used for the 
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

# MAGIC %md
# MAGIC ### Quick Check: Databricks and Constants
# MAGIC * Databricks (the [cloud-compute platform](https://databricks.com/) you're using now) allows you to write "notebooks" or collections of text and code in "cells" 
# MAGIC * Cells run sequentially and variables / settings persist within a notebook - [check here for a walk through](https://docs.microsoft.com/en-us/azure/databricks/notebooks/notebooks-use)
# MAGIC * Within this workshop, we'll use python and also use several helper scripts that are excuted like this...
# MAGIC ```
# MAGIC   %run ../features/location_ops
# MAGIC ```
# MAGIC * When exexcuting another notebook with the `%run` command, it's like runnning all of that code in the same notebook.

# COMMAND ----------

# MAGIC %run ../features/location_ops

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploring our source data... (exercise 2)
# MAGIC * This data comes from a 2009 data dump from [NYC Taxi and Limousine service](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
# MAGIC * We're loading our data from an ADLS source defined in our settings file, but you can upload your own via the [Blackbird self-service tool](https://blackbird.SERVICE_SITE/).
# MAGIC      * There are even instructions about [loading a CSV sheet](https://data.DOMAIN/resources/knowledge-base/28918666-8b4e-11ec-9463-0f9d673e50a5) (Excel friendly)
# MAGIC      * There are also instructions for a [graphic interface to file uploads](https://data.DOMAIN/resources/knowledge-base/b03da5a8-8e83-11ec-a22b-4d4634dc58c6)
# MAGIC * There are a few types of data columns...
# MAGIC    * *"key"* columns (for both pickup and dropoff times)
# MAGIC      * `pickup_datetime` and `dropoff_datetime` - a timestamp for pickup and dropoff
# MAGIC      * `pickup_longitude` and `pickup_latitude` - location information for a pickup
# MAGIC    * *"data"* columns
# MAGIC      * `passenger_count`, `total_amount` (and others) - that describe the rides themselves
# MAGIC * Finally, note that for speed we're sub-sampling to just 10% of our source data because we're still experimenting.
# MAGIC    * **Tip:** Subsampling can come in very handy to speed up development in the early stages of data exploration.
# MAGIC      
# MAGIC **Extra:** For the experienced users, if you're curious about the spatial encoding we're using, 
# MAGIC it's [Uber's H3 library](https://towardsdatascience.com/uber-h3-for-data-analysis-with-python-1e54acdcc908). 
# MAGIC Check out the settings file in this workshop for a discussion of the h3 resolution we chose.

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

# load our sampled data (NYC taxi paths)
path_read = CREDENTIALS['paths']['nyctaxi_h3_sampled']

if CREDENTIALS['constants']['EXPERIENCED_MODE']:  # experienced mode shows how the sampling is done
    path_raw = CREDENTIALS['paths']['nyctaxi_raw']
    # don't forget to apply our initial data filter
    df_taxi_raw = dataset_filter_raw(spark.read.format('delta').load(path_raw))
    if not CREDENTIALS['constants']['WORKSHOP_ADMIN_MODE']:   # subsample if not admin (it's a lot of data)
        df_taxi_raw = df_taxi_raw.sample(CREDENTIALS['constants']['DATA_SUBSAMPLE_RATIO'], seed=42)
    df_taxi_encoded = point_encode_h3(df_taxi_raw, 'dropoff_latitude', 'dropoff_longitude', 
                                      CREDENTIALS['constants']['RESOLUTION_H3'], 'dropoff_h3')
    df_taxi_encoded = point_encode_h3(df_taxi_encoded, 'pickup_latitude', 'pickup_longitude', 
                                      CREDENTIALS['constants']['RESOLUTION_H3'], 'pickup_h3')
    
    if CREDENTIALS['constants']['WORKSHOP_ADMIN_MODE']:   # if admin, we'll rewrite for the main data copy
        path_full = CREDENTIALS['paths']['nyctaxi_h3']
        path_read = CREDENTIALS['paths']['nyctaxi_h3_sampled']
        dbutils.fs.rm(path_full, True)
        df_taxi_encoded.write.format('delta').save(path_full)
        df_taxi_encoded = spark.read.format('delta').load(path_full)  # this trick clears the execute stack in spark
        df_taxi_encoded = df_taxi_encoded.sample(CREDENTIALS['constants']['DATA_SUBSAMPLE_RATIO'], seed=42)  
        dbutils.fs.rm(path_read, True)
        df_taxi_encoded.write.format('delta').save(path_read)   # this trick clears the execute stack in spark

df_taxi_encoded = spark.read.format('delta').load(path_read)  # read data with geo h3 tag

fn_log(f"Taxi data columns: {df_taxi_encoded.columns}")
display(df_taxi_encoded)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's map to a region for ML/AI Enrichment!
# MAGIC * Okay, now that we have our samples and some undertanding of where they happened, let's map
# MAGIC   them to regions that make sense.
# MAGIC * What location or geographies spring to mind....?
# MAGIC   * Zip codes - **Pro:** mailing zip codes are used all over, **Con:** may be too large of an area?
# MAGIC   * buildings - **Pro:** specific points of interest in a location, **Con:** May not be enough of them
# MAGIC   * citys and states - **Con:** not small enough for our modeling 
# MAGIC * ZIP codes are pretty good, right? Let's start there!
# MAGIC * [Image source](https://www.pexels.com/photo/smartphone-car-technology-phone-33488/)
# MAGIC 
# MAGIC <img src='https://images.pexels.com/photos/33488/navigation-car-drive-road.jpg?auto=compress&cs=tinysrgb&w=640&h=427&dpr=2' width='300px' title="predicting impacted locations" />

# COMMAND ----------

# load geometry for zip codes and filter for NEW YORK state; 
df_shape_zip = (spark.read.format('delta').load(CREDENTIALS['paths']['geometry_zip'])
    .filter(F.col('state')==F.lit('NY'))
)
df_zip_cells = shape_encode_h3cells(df_shape_zip, ['zip'], CREDENTIALS['constants']['RESOLUTION_H3'], 'geometry')

# load geometry for NEW YORK state; convert to geometry prsentation format
pdf_shape_states = (spark.read.format('delta').load(CREDENTIALS['paths']['geometry_state'])
    .filter(F.col('stusps')==F.lit('NY'))
    .toPandas()
)
pdf_shape_states['geometry'] = pdf_shape_states['geometry'].apply(lambda x: wkt.loads(x))

# load geometry for NEW YORK state; convert to geometry prsentation format
pdf_shape_states = (spark.read.format('delta').load(CREDENTIALS['paths']['geometry_state'])
    .filter(F.col('stusps')==F.lit('NY'))
    .toPandas()
)
pdf_shape_states['geometry'] = pdf_shape_states['geometry'].apply(lambda x: wkt.loads(x))

# join for both the pick-up and drop off
df_taxi_indexed = (
    point_intersect_h3cells(df_taxi_encoded.withColumnRenamed('pickup_h3', 'h3'),   # temp rename
                            'pickup_latitude', 'pickup_longitude', 
                            CREDENTIALS['constants']['RESOLUTION_H3'], df_zip_cells, col_h3='h3')
    .withColumnRenamed('zip', 'pickup_zip')   # rename merged zip
    .withColumnRenamed('h3', 'pickup_h3').drop('geoid')   # fix rename and drop extra
)
df_taxi_indexed = (
    point_intersect_h3cells(df_taxi_indexed.withColumnRenamed('dropoff_h3', 'h3'),   # temp rename
                            'dropoff_latitude', 'dropoff_longitude', 
                            CREDENTIALS['constants']['RESOLUTION_H3'], df_zip_cells, col_h3='h3')
    .withColumnRenamed('zip', 'dropoff_zip')   # rename merged zip
    .withColumnRenamed('h3', 'dropoff_h3').drop('geoid')   # fix rename and drop extra
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Flex your CDS muscle!
# MAGIC Remember, to be a CDS, one doesn't need to know how to program or train ML models to be a **CDS**!  Let's drive that point home.
# MAGIC * What we have so far is a set of data (our taxi records) that we've joined to zip codes.
# MAGIC * So we've enhanced our initial dataset with a spatial record, which is part of our "NYC Taxi 2035" goal.
# MAGIC * One easy thing to do is look at the stats to see if there are any homeruns for zip codes that should be prioritized.
# MAGIC   * To get there, we'll do two things: group by zip code and count those hits
# MAGIC   * Visualize our results to make sure it looks reasonable (there's helper code for that)

# COMMAND ----------

path_read = CREDENTIALS['paths']['nyctaxi_h3_zips']

# only admins write this one (it takes almost 10m to aggregate)
if CREDENTIALS['constants']['EXPERIENCED_MODE'] and CREDENTIALS['constants']['WORKSHOP_ADMIN_MODE']:  
    # now filter to relevant zips (those found in our taxi data)
    df_zips = (df_taxi_indexed
        .select(F.col('pickup_zip').alias('zip'))   # first sum by pickup
        .groupBy('zip').agg(F.count('zip').alias('count'))
        .union(df_taxi_indexed.select(F.col('dropoff_zip').alias('zip'))   # and then by dropoff
            .groupBy('zip').agg(F.count('zip').alias('count'))
        )
        .groupBy('zip').agg(F.sum('count').alias('count'))  # sum them all together
        .withColumnRenamed('zip', '_zip_join')
    )
    df_zips = (df_zips
        .join(df_shape_zip, df_zips['_zip_join']==df_shape_zip['zip'], 'inner')
        .drop('_zip_join')
        .withColumn('count_log10', F.log(10.0, F.col('count')))
    )
    dbutils.fs.rm(path_read, True)
    df_zips.write.format('delta').save(path_read)

    
fn_log("Grouping into a plottable dataframe...")
df_zips = spark.read.format('delta').load(path_read)
pdf_sub = df_zips.toPandas().sort_values(by='count', ascending=False)
pdf_sub['geometry'] = pdf_sub['geometry'].apply(lambda x: wkt.loads(x))
num_total = len(pdf_sub['count'])
shape_plot_map(pdf_sub, col_viz='count_log10', txt_title=f"Zipcode Log-Count ({num_total} total zips)", 
               gdf_background=pdf_shape_states)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Quiz Time!
# MAGIC Okay, what do you as CDSs think that we can do to improve this data?
# MAGIC 
# MAGIC 1. Filter by costs
# MAGIC 2. Filter by zip code
# MAGIC 3. Filter by distance
# MAGIC 4. Just start over!

# COMMAND ----------

# some initial rules to filter zip codes by location
pdf_sub_part = pdf_sub[(pdf_sub['intptlon'] > -74.5) & (pdf_sub['intptlon'] < -73.5) & (pdf_sub['intptlat'] < 41.1)]
num_total = len(pdf_sub_part['count'])
list_cities = pdf_sub_part['city'].unique()
fn_log(f"Cities ({len(list_cities)} unique): {list_cities} ")
shape_plot_map(pdf_sub_part, col_viz='count_log10', txt_title=f"Zipcode Log-Count ({num_total} total zips, {len(list_cities)} total cities)", 
               gdf_background=pdf_shape_states)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Know when to fold 'em
# MAGIC Zip code isolation and geospatial filtering wasn't quite enough.
# MAGIC 1. Locations go far beyond Manhattan
# MAGIC 2. Low-frequency areas (log view from above) like NY state, parts of Long Island and Staten Island still appear
# MAGIC 
# MAGIC So it's time to look around at what other 
# MAGIC data may be available.  Luckily, NYC's data transparency pays off again and it looks like we have 
# MAGIC records of the NYC T&LC (Taxi & Limousine Commission) zones.
# MAGIC 
# MAGIC <img src='https://data.cityofnewyork.us/api/assets/3FF54443-CD9C-4E56-8A20-8D2BD245BD1A?nyclogo300.png' width='300px' title="predicting impacted locations" />
# MAGIC 
# MAGIC Let's grab the shape definitions from this site and try geo identificaiton one more time.
# MAGIC 
# MAGIC * [Image source](https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/d3c5-ddgc)

# COMMAND ----------

# load geometry for zip codes and filter for NEW YORK state; 
df_shape_tlc = spark.read.format('delta').load(CREDENTIALS['paths']['geometry_nyctaxi'])
display(df_shape_tlc)

df_tlc_cells = shape_encode_h3cells(df_shape_tlc, ['zone'], CREDENTIALS['constants']['RESOLUTION_H3'], 'the_geom')

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
# MAGIC ## Data Exploration Conclusions
# MAGIC Not too bad, right?  We've come to a few decisions about how to represent our data....
# MAGIC 
# MAGIC * Group into zones as defined by the NYC T&LC by location
# MAGIC * Incorporating external data can add some SME (subject matter expert) spice to our modeling
# MAGIC 
# MAGIC We know about ride volume, but let's dig into knowing about our customers as well.

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploring Customer Data Joins (exercise 3)
# MAGIC 
# MAGIC Now that we have awareness of how to group our data, let's link it to customers.
# MAGIC * Within the company, there are lots of extra data that can add insights to these explorations.
# MAGIC   * Account information - tenure, pricing models, active lines and data
# MAGIC   * Demographics - age, household income, ethnic makeup
# MAGIC   * Behavioral data - phone calls, browser visits, etc.
# MAGIC   
# MAGIC It's important to identify the right way to expand your data.  Which do you think would be most helpful?
# MAGIC   * Account
# MAGIC   * Demographics
# MAGIC   * Behavioral
# MAGIC   * something else?

# COMMAND ----------

# MAGIC %run ../features/featurestore_ops

# COMMAND ----------

path_read = CREDENTIALS['paths']['demographics_raw']
# load geometry for zip codes and filter for NEW YORK state; 
df_shape_tlc = spark.read.format('delta').load(CREDENTIALS['paths']['geometry_nyctaxi'])

# feel free to try this at home, but you DO have to set up some other secrets / codes to use the feature store...
#    see - https://data.DOMAIN/products-and-services/self-service/Atlantis-Feature-Store
if CREDENTIALS['constants']['EXPERIENCED_MODE'] and CREDENTIALS['constants']['WORKSHOP_ADMIN_MODE']:
    # load raw features from the feature store (this is an *advanced* topic that we'll review in hour 3!)
    df_demos_raw = demographic_load_fs()

    # load geometry for zip codes and filter for NEW YORK state; 
    df_tlc_cells = shape_encode_h3cells(df_shape_tlc, ['zone'], CREDENTIALS['constants']['RESOLUTION_H3'], 'the_geom')

    # join for both the pick-up and drop off
    df_demos_indexed = (
        point_intersect_h3cells(df_demos_raw, 'latitude_cif', 'longitude_cif', 
                                CREDENTIALS['constants']['RESOLUTION_H3'], df_tlc_cells, col_h3='h3')
        .dropna(subset=['zone'])    # drop those that are null (that means they're not in our study area)
        .drop('latitude_cif', 'longitude_cif')    # drop the raw lat/long to avoid SPI/PII danger
        .withColumnRenamed('zone', '_zone_match')   # rename for next matching below
    )
    df_demos_indexed = (df_demos_indexed   # also going to add borough name!
        .join(df_shape_tlc.select('zone', 'borough'),
              df_shape_tlc['zone']==df_demos_indexed['_zone_match'], 'left')
        .drop('_zone_match')
    )
    dbutils.fs.rm(path_read, True)
    df_demos_indexed.write.format('delta').save(path_read)  # save new demographics subset

fn_log("Sneak peak at demographics...")
df_demos_indexed = spark.read.format('delta').load(path_read)
display(df_demos_indexed.limit(10))

# pdf_sub = df_zones.toPandas().sort_values(by='count', ascending=False)
# pdf_sub['geometry'] = pdf_sub['geometry'].apply(lambda x: wkt.loads(x))
# num_total = len(pdf_sub['count'])
# shape_plot_map(pdf_sub, col_viz='count_log10', txt_title=f"Zone Log-Count ({num_total} total zones)", 
#                gdf_background=pdf_shape_states)



# COMMAND ----------

# MAGIC %md 
# MAGIC ## Demographic Exploration
# MAGIC Demographics are senstive, but in aggregate, they give a peek at the population that you're serving in an area.
# MAGIC From the table above, we have subset of columns (personal addresses and names have been removed) that describe 
# MAGIC several characterstics of the inhabitant. 
# MAGIC 
# MAGIC *NOTE: These demographics are sourced from an outside vendor so it's outside 
# MAGIC the scope of this workshop to focus on collection or class namings.*
# MAGIC 
# MAGIC In this notebook, we refer to these as 'demographic factors' and will experiment with joining various factors to
# MAGIC our source data.
# MAGIC * `gnrt` - the "generation" (by age) of the individual (e.g. "Millenials", "GenX", etc.)
# MAGIC * `ethnc_grp` - the ethnic group of the individual (e.g. "Asian", "Hispanic", etc.)
# MAGIC * `gndr` - the gender of the individual (e.g. "F", or "M")
# MAGIC * `edctn` - the education level of the individual (e.g. "Bach Degree - Likely", "Some College - Likely", etc.)
# MAGIC * `marital_status_cif` - marital status (e.g. "5M" or "5S")
# MAGIC * `hshld_incme_grp` - the group household income (e.g. "$150K-$249K", "<$50K", etc.)
# MAGIC 
# MAGIC For those techncially saavy individuals, we are **not** aggregating by a single household and are instead getting
# MAGIC numbers for everyone that lives in an area.  Numerically that means households that have multiple individuals 
# MAGIC get more weight, but this may only matter if comparing factors like *household income* (`hshld_incme_grp`).

# COMMAND ----------

path_read = CREDENTIALS['paths']['demographics_factors']

if CREDENTIALS['constants']['EXPERIENCED_MODE'] and CREDENTIALS['constants']['WORKSHOP_ADMIN_MODE']:
    df_demos_pivot = demographic_factors_count(df_demos_indexed, list_factors=['zone'])
    dbutils.fs.rm(path_read, True)
    df_demos_pivot.write.format('delta').save(path_read)  # save new demographics factorized version

fn_log("Factorized demographics by zone...")
df_demos_pivot = spark.read.format('delta').load(path_read)
list_factors = df_demos_pivot.select('factor').distinct().collect()
fn_log(f"These factors are available: {list_factors}")
display(df_demos_pivot)

# load geometry for nyc taxi zones
df_shape_tlc = (
    spark.read.format('delta').load(CREDENTIALS['paths']['geometry_nyctaxi'])
    .withColumnRenamed('zone', '_zone_join')
)

# now we'll join
pdf_demos_pivot_shape = (df_demos_pivot
    .join(df_shape_tlc.select('_zone_join', 'the_geom'), df_shape_tlc['_zone_join']==df_demos_pivot['zone'])
    .drop('_zone_join')
    .withColumn('count_log10', F.log(10.0, F.col('count')))
    .withColumnRenamed('the_geom', 'geometry')
    .toPandas()
)
pdf_demos_pivot_shape['geometry'] = pdf_demos_pivot_shape['geometry'].apply(lambda x: wkt.loads(x))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Visual Inspection of Distributions
# MAGIC As a CDS, a lot can be accomplished by simply looking at data distributions for patterns.  We'll do that below
# MAGIC by first inspecting the alignment of our previous map (showing the number of rides by T&LC zones) and comparing
# MAGIC it to a few demographic factors (showing the population by T&LC zones).

# COMMAND ----------

pdf_sub = pdf_demos_pivot_shape[pdf_demos_pivot_shape['factor']=='gnrt']
shape_plot_map_factors(pdf_sub, col_factor='value', col_viz='count_log10', use_log=True,
                       txt_title=f"Age Difference by Zone (%)", col_norm='zone',
                       gdf_background=pdf_shape_states)

# COMMAND ----------

# MAGIC %md
# MAGIC From the plots above (first looking at "generation" distribtution), we notice a few things.
# MAGIC 1. The distribution of populations may vary in magnitude (e.g. the numbers on 
# MAGIC    legend are different), but the distributions themselves are roughly the same in relative intensity
# MAGIC    for each generation group.
# MAGIC 2. There is one pocket for "Millenials" in Brooklyn and generally "Seniors" have neutral or lower overall counts.
# MAGIC 3. Looking at the maps, there aren't many significant differences in which areas are "inactive" or
# MAGIC    simply have too few individuals of that demographic factor.  
# MAGIC       * For example, many of the zones in Queens and Brooklyn are active across all generations.

# COMMAND ----------

pdf_sub = pdf_demos_pivot_shape[pdf_demos_pivot_shape['factor']=='ethnc_grp']
shape_plot_map_factors(pdf_sub, col_factor='value', col_viz='count_log10', use_log=True,
                       txt_title=f"Ethnicity Difference by Zone (%)", col_norm='zone',
                       gdf_background=pdf_shape_states)


# COMMAND ----------

# MAGIC %md
# MAGIC From the plots above there is a different story when comparing ethnicity.
# MAGIC 1. There are clear preferences for certain groups by population count.
# MAGIC    * *African American* in Queens, 
# MAGIC    * *Hispanic* in Bronx
# MAGIC    * *General* = Staten Island + Manhattan + Brooklyn, 
# MAGIC 2. There are serious disparities by zone to be considered when analyzing our historical rides map

# COMMAND ----------

pdf_sub = pdf_demos_pivot_shape[pdf_demos_pivot_shape['factor']=='hshld_incme_grp']
shape_plot_map_factors(pdf_sub, col_factor='value', col_viz='count_log10', use_log=True,
                       txt_title=f"HH Income Difference by Zone (%)", col_norm='zone',
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

# MAGIC %md
# MAGIC ## Check to see unique zip codes
# MAGIC Uh oh, this includes all of NY state and that's not what we want

# COMMAND ----------

# MAGIC %md
# MAGIC ## Now, join other data from taxi shape
# MAGIC This will limit those zip codes that map to NYC taxi service, not the greater state...

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
