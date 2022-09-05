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

# MAGIC %run ../features/location_ops

# COMMAND ----------

# clear widgets first
dbutils.widgets.removeAll()

# create model map
list_model_map = [['00 Raw Demographics', None, 'Difference'], 
                  ['01 Raw Fare Total', 'nyctaxi_h3_zones', 'Historical Rides']]
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
if model_sel_last != model_sel:
    model_path = [x[1] for x in list_model_map if x[0]==model_sel][0]  # find the path part of selected model
    model_sel_viz = [x[2] for x in list_model_map if x[0]==model_sel][0]  # find the graph title part
    model_sel_last = model_sel
    if model_path is None:  # if no ride count, just create sample equal in all ones
        df_rides = df_demos_pivot_all.select('zone').withColumn('_count_rides', F.lit(1))
    else:
        path_read = CREDENTIALS['paths'][model_path]
        df_rides = (spark.read.format('delta').load(path_read)
            .select('zone', F.col('count').alias('_count_rides'))
        )
    # recompute the demo-based counts by ride scalar
    df_demos_pivot = (df_demos_pivot_all
        .join(df_rides, ['zone'])
        .withColumnRenamed('count', '_count_demos')
        .withColumn('count', F.col('_count_demos') * F.col('_count_rides'))
    )

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
del pdf_plot_demo['the_geom']

shape_plot_map_factors(pdf_plot_demo, col_factor='value', col_viz='count_log10', use_log=True,
                       txt_title=f"{demo_sel} {model_sel_viz} by Zone (%)", col_norm='zone',
                       gdf_background=pdf_shape_states)

# COMMAND ----------


