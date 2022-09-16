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
# MAGIC # Temporarily Under Construction
# MAGIC <img src='https://images.pexels.com/photos/211122/pexels-photo-211122.jpeg?cs=srgb&dl=pexels-fernando-arcos-211122.jpg&fm=jpg&w=640&h=426' width='300px' title="Data Under Construction" />
# MAGIC 
# MAGIC Sorry, with some last-minute discoveries of data errors, we need a little more time in cleaning up the data and scripts in this section.  We'll be online before the workshop!

# COMMAND ----------

str_quit = f"Notebook temporarily disabled to fix some final data issues, please come back again!"
dbutils.notebook.exit(str_quit)

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

# MAGIC %run ../examples/calibrated_equalized_odds

# COMMAND ----------

# MAGIC %md
# MAGIC ## n3.e8 Auxiliary Feature Creation

# COMMAND ----------



# COMMAND ----------

import scipy.stats as st   # zscore norm

# col_disparity_ind = 'needs_mitigation'
# factor_active = 'ethnc_grp'

# path_read = CREDENTIALS['paths']['demographics_aux']

# # defer to admin compute, takes ~5m each
# if CREDENTIALS['constants']['EXPERIENCED_MODE'] and CREDENTIALS['constants']['WORKSHOP_ADMIN_MODE']:
#     udf_zscore_to_percent = F.udf(lambda x: float(st.norm.cdf(x)))
#     df_demos_aux = (
#         taxi_zone_demos(factor_active)   # compute column-wise z-score for a demographic factor
#         .join(spark.read.format('delta').load(CREDENTIALS['paths']['demographics_disparity_base'])
#               .filter(F.col('factor') == F.lit(factor_active))
#               .groupBy('zone').agg(
#                   F.min('overall_quantile').alias('overall_quantile'),   # retain quantile scores
#                   F.max('cnt_zscore').alias('zscore_max'),   # this will tell us how much to weight the sample
#               )
#               .withColumn(col_disparity_ind, F.when(F.col('overall_quantile')==F.lit(1), F.lit(1)).otherwise(F.lit(0)))
#               .withColumn('z_percent', udf_zscore_to_percent(F.col('zscore_max')) )   # convert zscore -> [0, 1]
#               .withColumn('sample_weight', F.lit(0.5) + (F.col(col_disparity_ind)*F.col('z_percent')) )   # weight is [0.5 - percent/2]
#               , ['zone'])  # use those in top quantile as positive label, join by zone
#         .drop('zscore_max')
#     )
#     dbutils.fs.rm(path_read)
#     df_demos_aux.write.format('delta').save(path_read)
    
# df_demos_aux = spark.read.format('delta').load(path_read).orderBy('zone')
# display(df_demos_aux)

# COMMAND ----------

# group by date, zone
path_read = CREDENTIALS['paths']['nyctaxi_geo_labeled']

# use the raw training data
df_train = (
    spark.read.format('delta').load(path_read)   # save on compute
    .filter(F.col('dataset')==F.lit('train'))   # filter for training data
    .withColumnRenamed('pickup_zone', 'zone')
    .join(df_demos_aux, ['zone'])  # join only aux demo features
    .drop('overall_quantile')
)
display(df_train)

df_test = (
    spark.read.format('delta').load(path_read)   # save on compute
    .filter(F.col('dataset')==F.lit('test'))   # filter for training data
    .withColumnRenamed('pickup_zone', 'zone')
    .join(df_demos_aux, ['zone'])  # join only aux demo features
    .drop('overall_quantile')
)

# JOIN AGAINST AUX DATA
# TRANSFORM THE 'needs_mitigation' label into a weight



# COMMAND ----------

# col_label, list_features = taxi_train_columns()

# list_out = [df_train.colRegex(f"`({factor_active}.*)`")]
# print(list_out)

# COMMAND ----------

# col_label, list_features = taxi_train_columns()

# list_feats_aux += [df_train.colRegex(f"`^{factor_active}.*`")]

    
# # pull out our test data from the labeled dataset
# path_read = CREDENTIALS['paths']['nyctaxi_geo_labeled']
# df_labeled = spark.read.format('delta').load(path_read)
# df_test = df_labeled.filter(F.col('dataset')==F.lit('test'))

# experienced mode derivation of data (about 15-25m)
#    this code will actually train a new model based on the input `list_features` using the label `col_label`
#    this code should be generic enough for you to reuse in your own 
if CREDENTIALS['constants']['EXPERIENCED_MODE'] and CREDENTIALS['constants']['WORKSHOP_ADMIN_MODE']:
    col_label, list_features = taxi_train_columns()

    # comprehensive check for impact by data size
    # list_ratio_test = [0.05, 0.5, 0.8]
    list_ratio_test = [0.05]   # we tried several sample ratios and they were all about the same
    for sample_ratio in list_ratio_test:
        df_train = (
            df_labeled.filter(F.col('dataset')==F.lit('train'))
            .select(list_features+[col_label])
            .sample(sample_ratio, seed=42)
        )
        # the first line searches for best parameters (set num_folds=1 to skip the search)
        pipeline, best_hyperparam = modeling_gridsearch(df_train, col_label, num_folds=3)
        # these lines actually train the model
        best_hyperparam['training_fraction'] = sample_ratio
        run_name, df_pred = modeling_train(df_train, df_test, col_label, "taxi_popular", 
                                           pipeline, best_hyperparam, list_inputs=list_features)

        
        


# COMMAND ----------

# MAGIC %md
# MAGIC ## Backup Mitigation Work
# MAGIC Unfortunately, we couldn't quite get the AIF360 library to work correctly with our data.  So for now,
# MAGIC pieces of code are frozen below, but are not used in our evaluation.

# COMMAND ----------

if False:   # apply classification
    # group by date, zone
    path_read = CREDENTIALS['paths']['nyctaxi_geo_labeled']
    df_labeled = spark.read.format('delta').load(path_read)   # save on compute

    # pull out our test data from the labeled dataset
    df_valid = df_labeled.filter(F.col('dataset')==F.lit('validate'))
    df_valid_pred = model_predict(df_valid, "taxi_popular")

col_disparity_ind = 'needs_mitigation'
col_disparity_type = 'needs_mitigation_value'
factor_active = 'ethnc_grp'
class_privilaged = '(privileged)'
df_valid_pred_debias = None

if False:
    df_debias_zone = (
        spark.read.format('delta').load(CREDENTIALS['paths']['demographics_disparity_base'])
        .filter(F.col('factor')==F.lit(factor_active))   # limit to current factor
        .withColumn(col_disparity_ind, F.when(F.col('overall_quantile')==F.lit(1), F.lit(1)).otherwise(0))   # generate binary
        .withColumn(col_disparity_type, F.when(F.col('overall_quantile')==F.lit(1), F.lit(F.col('value'))).otherwise(class_privilaged))   # retain souce
        .select('zone', col_disparity_ind, col_disparity_type)  # keep cols of interest
        .orderBy(F.col(col_disparity_ind).desc())  # sort for de-dup
        .dropDuplicates(['zone'])
    )

    df_valid_pred_debias = (df_valid_pred
        .select('is_top_probability', 'is_top_predict', 'is_top', 'pickup_zone')
        .withColumnRenamed('pickup_zone', 'zone')
        .join(df_debias_zone, ['zone'])
        #.orderBy(F.col(col_disparity_ind).desc())
                            
    )                  
    pdf_valid_pred = (df_valid_pred_debias                 
        .toPandas()
        .set_index(col_disparity_type)
    )
    display(df_valid_pred_debias)

if df_valid_pred_debias is not None:
    # pdf_valid_pred = pdf_valid_pred.set_index(col_disparity_type)
    print(pdf_valid_pred)
    par_gap = statistical_parity_difference(pdf_valid_pred['is_top'], pdf_valid_pred['is_top_predict'], 
                                            prot_attr=col_disparity_type, priv_group=class_privilaged, pos_label=1)

    #     score = balanced_accuracy_score(pdf_train.set_index(col_protected)['is_top'], pdf_train['is_top_probability'].astype('int'))
    fn_log(f"Parity gap equals {par_gap:.2%}")


# COMMAND ----------

from pyspark.ml.feature import StringIndexer
if False:
    # group by date, zone
    path_read = CREDENTIALS['paths']['nyctaxi_geo_labeled']
    df_labeled = spark.read.format('delta').load(path_read)   # save on compute

    # pull out our test data from the labeled dataset
    df_valid = df_labeled.filter(F.col('dataset')==F.lit('validate'))
    df_valid_pred = model_predict(df_valid, "taxi_popular")
    # display(df_valid_pred)

    # join the aux features with our predictions
    col_id = "_uni_id"
    df_demos_mitigate = (df_valid_pred.sample(CREDENTIALS['constants']['DATA_SUBSAMPLE_RATIO'])
        .withColumnRenamed('pickup_zone', 'zone')
        .select('zone', 'is_top', 'is_top_probability')
        .join(df_demos_aux, ['zone'])
        .withColumn(col_id, F.monotonically_increasing_id())
        .fillna(0)
    )
    # get a transformer to convert text zone into numeric feature
    trans_zone = StringIndexer(inputCol='zone', outputCol='zone_id').fit(df_demos_mitigate)
    df_demos_mitigate = trans_zone.transform(df_demos_mitigate)

    # what's it look like after?
    display(df_demos_mitigate)
    
    # split to train/test on validation
    from sklearn.model_selection import train_test_split
    pdf_samples = df_demos_mitigate.drop('zone').toPandas()
    pdf_train, pdf_test = train_test_split(pdf_samples, test_size=.3)
    print(pdf_train.columns)

    # pdf_reset = pdf_train.set_index(col_protected, drop=False)
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    score = discrimination(pdf_train.set_index(col_protected)['is_top'], pdf_train['is_top_probability'], prot_attr=col_protected)
    print(score)
    score = balanced_accuracy_score(pdf_train.set_index(col_protected)['is_top'], pdf_train['is_top_probability'].astype('int'))
    print(score)

    # consider investigating more - https://github.com/Trusted-AI/AIF360/blob/master/examples/sklearn/demo_fairadapt_sklearn.ipynb
    # pdf_train = df_demos_mitigate.sample(0.6)
    # pdf_test = df_demos_mitigate.join(df_train, ['zone'], 'leftanti')
    col_label = 'is_top'
    col_protected = 'needs_mitigation'

    # from aif360.sklearn.postprocessing import CalibratedEqualizedOdds
    pp = CalibratedEqualizedOdds(col_protected, cost_constraint='weighted', random_state=42)
    # ceo = PostProcessingMeta(estimator=lr, postprocessor=pp, random_state=42)
    pp.fit(pdf_train, pdf_train[[col_protected]], col_label)
    y_pred_zone = pp.predict(pdf_test)
    y_proba_zone = pp.predict_proba(pdf_test)
    display(y_proba_zone)
    
    # df_disparity = taxi_zone_demo_disparities('hshld_incme_grp', quantile_size=10)
    # df_disparity = taxi_zone_demo_disparities('ethnc_grp', quantile_size=10)

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
# predict the locations that will be most profitable morning / afternoon / evening on days of week
# generate model that predicts by areas

# overlay biggest demographic classes by age or race or income
# compare disparities for evening pickups

# solution 1 - create model to better sample
# solution 2- postfix to hit min score
