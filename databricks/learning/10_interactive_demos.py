# Databricks notebook source
# 
# This file contains visualization work used for the 
#   2022 Machine Learning Workshop (part of the Software Symposium)!
#   https://FORWARD_SITE/mlworkshop2022 
#      OR https://INFO_SITE/cdo/events/internal-events/4354c5db-3d3d-4481-97c4-8ad8f12686f1
#
# You can (and should) change them for your own experiments, but they are uniquely defined
# here for constants that we will use together.


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Interactive Demographic Implications
# MAGIC Using aggregated demographics, this notebook provides a single place to look at the imapct of several models
# MAGIC by taxi zone.  This notebook is meant as a companion for the previously execited notebooks that build
# MAGIC an understanding of geography, demographics, and model building.
# MAGIC 
# MAGIC This notebook uses [Databricks widgets](https://docs.databricks.com/notebooks/widgets.html) to allow a semi-interactive
# MAGIC exploration of models and precompted demographics.
# MAGIC 
# MAGIC * ***demographic*** - Using a subset of demographics that are aggregated by NYC T&LC zone, display the impact
# MAGIC   of the specified model on that zone. 
# MAGIC * ***model*** - Coupled to modeling choses in a notebook, various models are used to compute whether or not a ride
# MAGIC   is recommended.  These recommendations are applied to a test dataset and the resultant descisions are used
# MAGIC   to compute a disparity (more rides or less rides as a [z-score](https://en.wikipedia.org/wiki/Standard_score)) 
# MAGIC   versus all other zones in the area.

# COMMAND ----------

# MAGIC %run ../features/location_ops

# COMMAND ----------

# clear widgets first
dbutils.widgets.removeAll()

# create model map
list_model_map = [['Raw Demographics (n1.e5)', None, 'Difference'], 
                  ['Refined Fare Total (n2.e6)', 'nyctaxi_h3_historical', 'Ride Disparity'],
                  ['Learned Fare Total (n2.e7)', 'nyctaxi_h3_learn_base', 'Ride Disparity'],
                 ]
model_sel_last = None
dbutils.widgets.dropdown("model", list_model_map[0][0], [x[0] for x in list_model_map])

# load specific demographic features
path_read = CREDENTIALS['paths']['demographics_factors']

dict_map = [[0, 'ethnc_grp', 'Ethnicity'], [1, 'hshld_incme_grp', "HH Income"], [2, 'gnrt', "Age"], 
            [3, 'marital_status_cif', "Marital Status"], [4, 'edctn', 'Education'], 
            [5, 'ethnic_sub_grp', 'Ethnic Sub Group'], [6, 'gndr', 'Gender']]
df_mapping = spark.createDataFrame(pd.DataFrame(dict_map, columns=['priority', 'factor', 'factor_name']))

fn_log("Loading factorized demographics by zone...")
df_demos_pivot_all = spark.read.format('delta').load(path_read).join(df_mapping, ['factor'])
list_factors = df_demos_pivot_all.select('factor').distinct().toPandas()

# load geometry for nyc taxi zones
df_shape_tlc = spark.read.format('delta').load(CREDENTIALS['paths']['geometry_nyctaxi'])
# pdf_shape_tlc = df_shape_tlc.toPandas()
# pdf_shape_tlc['geometry'] = pdf_shape_tlc['the_geom'].apply(lambda x: wkt.loads(x))

# load geometry for NEW YORK state; convert to geometry presentation format
pdf_shape_states = (spark.read.format('delta').load(CREDENTIALS['paths']['geometry_state'])
    .filter(F.col('stusps')==F.lit('NY')) 
    .toPandas()
)
pdf_shape_states['geometry'] = pdf_shape_states['geometry'].apply(lambda x: wkt.loads(x))

# make demographic drop down
list_demos = list(df_mapping.orderBy(F.col('priority')).select('factor_name').toPandas()['factor_name'])
dbutils.widgets.dropdown("demographic", list_demos[0], list_demos)



# COMMAND ----------

# update ride count by the currently selected model
model_sel = dbutils.widgets.get("model")
if model_sel_last != model_sel:   # avoid recompute if model didn't chage
    model_path = [x[1] for x in list_model_map if x[0]==model_sel][0]  # find the path part of selected model
    model_sel_viz = [x[2] for x in list_model_map if x[0]==model_sel][0]  # find the graph title part
    fn_log(f"Updating to new model {model_sel}... ({model_path})")
    model_sel_last = model_sel
    if model_path is None:  # if no ride count, just create zero z-score for disparity
        df_rides = df_demos_pivot_all.select('zone').withColumn('rides_z', F.lit(0))
    else:   # otherwise, compute z-score for disparity measure
        path_read = CREDENTIALS['paths'][model_path]
        df_rides = spark.read.format('delta').load(path_read)
        # note that there is some column renaming to perform first...
        row_stat_rides = df_rides.select(F.mean('volume').alias('mean'), F.stddev('volume').alias('std')).collect()[0]
        df_rides = (df_rides
            .withColumnRenamed('pickup_zone', 'zone')
            .withColumn("rides_z", (F.col('volume') - F.lit(row_stat_rides['mean']))/F.lit(row_stat_rides['std']))
            .withColumnRenamed('volume', 'rides')
            .select('zone', 'rides', 'rides_z')
        )
    # recompute the demo-based counts by ride scalar
    df_demos_pivot = df_demos_pivot_all.join(df_rides, ['zone'])

# Complete for visilzuation    
demo_sel = dbutils.widgets.get("demographic")

# now we'll join against shapes to plot
pdf_plot_demo = (df_demos_pivot
    .filter(F.col('factor_name') == F.lit(demo_sel))
    .join(df_shape_tlc.select('zone', 'the_geom'), ['zone'])
    .withColumn('count_log10', F.log(10.0, F.col('count')))
    .toPandas()
)
pdf_plot_demo['geometry'] = pdf_plot_demo['the_geom'].apply(lambda x: wkt.loads(x))

shape_plot_map_factors(pdf_plot_demo, col_factor='value', col_viz='count_log10', use_log=True,
                       txt_title=f"{demo_sel} {model_sel_viz} by Zone (%)", col_norm='zone',
                       col_disparity='rides_z', gdf_background=pdf_shape_states)

# COMMAND ----------


