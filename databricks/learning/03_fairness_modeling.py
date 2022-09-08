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
# MAGIC # Let's *Fix* "NYC Taxi 2025"!
# MAGIC <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Kernel_Machine.svg/640px-Kernel_Machine.svg.png' width='400px' title="ML Model Boundary" />
# MAGIC 
# MAGIC 
# MAGIC ## Review
# MAGIC * In the first notebook (**01_explore_data**) we looked at NYC Taxi data and regional demographic data.
# MAGIC * In the second notebook (**02_predict_debiased**) we built a prediction model and confirmed bias does exist.
# MAGIC    * The model predicts only by NYC T&LC zones instead of daypart because that data wasn't available.
# MAGIC    * We identified ride disparity for the most populous zones according to ethnicity and income as targets to fix.
# MAGIC * In this notebook, we'll demonstrate a way to **mitigate** bias in our historical prediction model through the use
# MAGIC   of auxiliary databases (here demographics data).
# MAGIC * As a reminder, your code will default to the _'beginner'_ path.  As a testament to your _'expertness'_ you can 
# MAGIC   edit the configuration file (`utilities/settings.py`) to set this variable.  *(HINT: Look for EXPERIENCED_MODE)*
# MAGIC * [image source](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Kernel_Machine.svg/640px-Kernel_Machine.svg.png)

# COMMAND ----------

# MAGIC %run ../features/location_ops

# COMMAND ----------

# MAGIC %run ../features/taxi_dataset

# COMMAND ----------

# MAGIC %run ../features/modeling

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auxiliary Feature Creation

# COMMAND ----------

# df_disparity = taxi_zone_demo_disparities('hshld_incme_grp', quantile_size=10)
# df_disparity = taxi_zone_demo_disparities('ethnc_grp', quantile_size=10)



# df_grouped = (df_disparity
#     .filter(F.col('overall_quantile')==F.lit(1))
#     .groupBy('value').agg(
#         F.count('value').alias('count'),
#         # F.countDistinct('zone').alias('distinct'),
#     )
#     .orderBy(F.col('count').desc())
# )
# fn_log(f"Disparities Detected: {df_grouped.toPandas().to_dict()}")

# df_grouped.toPandas().plot.bar('value', 'count', grid=True)
# display(df_grouped)
# fn_log(f"Total Zones: {df_disparity.select(F.countDistinct('zone').alias('count')).collect()}")






# plot gains / losses for each demographic (choose top N?)
# (repeat) add pre-filtering by sampling, measure + plot gains
# (repeat) add post-filtering by bias, measure + plot gains
# (repeat) add pre-processing for feature adjustment by bias, measure + plot gains
# 

# COMMAND ----------

# group by date, zone
path_read = CREDENTIALS['paths']['nyctaxi_geo_labeled']
df_labeled = spark.read.format('delta').load(path_read)   # save on compute

# pull out our test data from the labeled dataset
df_valid = df_labeled.filter(F.col('dataset')==F.lit('validation'))
df_valid_pred = model_predict(df_valid, "taxi_popular")


# COMMAND ----------

# also write predicted, aggregated stats by the zone
# df_zone_predict = tax_postproc_volumes(df_predict)


# clobber historical file, rewrite
# dbutils.fs.rm(path_read, True)
# df_zone_predict.write.format('delta').save(path_read)




# COMMAND ----------

# MAGIC %run ../examples/calibrated_equalized_odds

# COMMAND ----------

# from aif360.sklearn.postprocessing import CalibratedEqualizedOdds
df_demos_pivot_all = taxi_zone_demos('ethnc_grp')

pp = CalibratedEqualizedOdds('race', cost_constraint='fnr', random_state=42)
# ceo = PostProcessingMeta(estimator=lr, postprocessor=pp, random_state=42)
pp.fit(X_train, y_train)
y_pred_CEO = pp.predict(X_test)
y_proba_CEO = pp.predict_proba(X_test)

# COMMAND ----------

# plot gains / losses for each demographic (choose top N?)
# (repeat) add pre-filtering by sampling, measure + plot gains
# (repeat) add post-filtering by bias, measure + plot gains
# (repeat) add pre-processing for feature adjustment by bias, measure + plot gains
# 

# COMMAND ----------

# from pyspark.ml.tuning import CrossValidatorModel
# cvModel = CrossValidatorModel.read().load('/users/ez2685/cvmodel')
# print(cvModel.extractParamMap())

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
