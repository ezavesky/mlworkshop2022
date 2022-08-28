# Databricks notebook source
# 
# This file contains demographic feature retrieval functions used for the 
#   2022 Machine Learning Workshop (part of the Software Symposium)!
#   https://FORWARD_SITE/mlworkshop2022 
#      OR https://INFO_SITE/cdo/events/internal-events/4354c5db-3d3d-4481-97c4-8ad8f12686f1
#
# You can (and should) change them for your own experiments, but they are uniquely defined
# here for constants that we will use together.


# COMMAND ----------

# MAGIC %run ../utilities/settings

# COMMAND ----------

import geopandas as gpd
import pyspark.sql.functions as F
import pandas as pd


# COMMAND ----------

# load demographics from neustar/feature store, dropping personally identifiable field (e.g. name, custloc, etc)
def demographic_load_fs(list_columns=['zipcd', 'age', 'hshld_income_cd', 'est_curr_home_value', 'exact_age', 'gnrt', 'lang', 'ethnc_grp', 'latitude_cif', 'longitude_cif', 'building_area_1_cif', 'lot_size_square_feet_cif', 'market_value_land_cif', 'market_value_improvement_cif', 'number_of_units_cif', 'number_of_bedrooms_cif', 'year_built_cif', 'tax_amount_cif', 'length_of_residence_cif', 'edctn', 'state_cif', 'city_cif']):
    from featurestore import Client
    import os
    os.environ['no_proxy'] = 'atlantis-feature-store.SERVICE_SITE'
    client = Client('atlantis-feature-store.SERVICE_SITE:8080')

    # Token='Enter your PAT token' --- see the configure file
    client.auth.set_auth_token(CREDENTIALS['credentials']['ATLANTIS'])

    project = client.projects.get('Neustar')
    fs = project.feature_sets.get('Neustar Demographics')
    data_reference = fs.retrieve()
    df = data_reference.as_spark_frame(spark)
    fn_log(f"[demographic_load_fs] Available columns... {df.columns}")

    # filter to specifically requested columns
    if list_columns is not None:
        return df.select(list_columns)
    return df

# testing functions to load/save demographics
if False: 
    path_write = CREDENTIALS['paths']['demographics_raw']
    dbutils.fs.rm(path_write, True)
    df_sub = demographic_load_fs().filter(F.col('state_cif') == F.lit('NY'))
    df_sub.write.format('delta').save(path_write)
    df_sub = spark.read.format('delta').load(path_write)

# COMMAND ----------


