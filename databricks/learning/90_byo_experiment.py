# Databricks notebook source
# 
# This file contains BYO experimental modeling work used for the 
#   2022 Machine Learning Workshop (part of the Software Symposium)!
#   https://FORWARD_SITE/mlworkshop2022 
#      OR https://INFO_SITE/cdo/events/internal-events/4354c5db-3d3d-4481-97c4-8ad8f12686f1
#
# You can (and should) change them for your own experiments, but they are uniquely defined
# here for constants that we will use together.


# COMMAND ----------

# MAGIC %md
# MAGIC # Bring Your Own Experiment
# MAGIC <img src='https://images.pexels.com/photos/3826579/pexels-photo-3826579.jpeg?auto=compress&cs=tinysrgb&w=640&h=427&dpr=2' width='200px' title="your own opportunity" />
# MAGIC 
# MAGIC ## Background
# MAGIC * One hope of this workshop was that you could bring your own data in for exprimentation, so here's the chance.
# MAGIC * Make sure you've used the [Blackbird self-service tool](https://blackbird.SERVICE_SITE/) to create your own storage location.
# MAGIC * Uploading data can happen either by going directly to the [Azure Portal](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-portal#upload-a-block-blob) (https://PORTAL) or by the [Storage Explorer](https://data.DOMAIN/resources/knowledge-base/b03da5a8-8e83-11ec-a22b-4d4634dc58c6) (separate Azure app)
# MAGIC * [Image source](https://www.pexels.com/photo/unrecognizable-black-man-catching-taxi-on-city-road-5648421/)

# COMMAND ----------

# MAGIC %run ../features/featurestore_ops

# COMMAND ----------

# load your custom dataframe from your own scratch storage or prod location
# NOTE 1: to join to a specific customer, you'll need a unique identifier (ban, cloc, etc)
# NOTE 2: don't forget to grab your own token from the feature store and put it into the utilities/settings file
df_custom_data = X

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Feature Store
# MAGIC Look to the "advanced" part of the workshop for more information, but one capability for data 
# MAGIC enrichment that CDO provides it the "feature store".  Secretly, we used an exceprt from it during 
# MAGIC the first notebook, **01_explore_data** when we brought in demographics data.
# MAGIC 
# MAGIC To that end, there are three example problems provided below that may match your data needs.
# MAGIC * `demographics` - demographic expansion by CLOC
# MAGIC * `churn` - adding churn features to an account model by CLOC
# MAGIC * ...

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

if False:   # demographics data!
    # feel free to try this at home, but you DO have to set up some other secrets / codes to use the feature store...
    #    see - https://data.DOMAIN/products-and-services/self-service/Atlantis-Feature-Store
    
    path_read = f"{CREDENTIALS['paths']['scratch_root_personal']}/custom_demographics"
    list_files = dbutils.fs.ls(path_read)
    if not list_files:   # guess that we didn't have anything written here yet...
        # load raw features from the feature store (this is an *advanced* topic that we'll review in hour 3!)
        df_demos_raw = demographic_load_fs()
        df_custom_enriched = (df_custom_data
            .sample(0.1, seed=42)    # remember always start small to test!
            .join(df_demos_raw, df_custom_data['cloc'] == df_demos_raw['cloc_cust_nm'])
        )
        display(df_custom_enriched.limit(10))   

        # what next? consider using our other helper function in features/featurestore_ops, 
        df_demos_pivot = demographic_factors_count(df_custom_enriched, list_factors=['my_custom_factor'])

        fn_log("Saving custom demographics data...")
        df_demos_pivot.write.format('delta').save(path_read)

    # perform normal read
    df_demos_pivot = spark.read.format('delta').load(path_read)


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Churn Exploration
# MAGIC Churn features are built from behaviors of a customer and may include browser or account activity aggregates.
# MAGIC 
# MAGIC The sample code below loads from [this feature store reference](https://atlantis.SERVICE_SITE/featurestore/#/view-feature/dabbler/voluntary%20churn%20model?page=1&level=all-features.
# MAGIC 
# MAGIC There are several churn sets available in the [feature store](https://atlantis.SERVICE_SITE/featurestore/), so you're encouraged to use the right set for you.

# COMMAND ----------

if False:   # churn data!
    # feel free to try this at home, but you DO have to set up some other secrets / codes to use the feature store...
    #    see - https://data.DOMAIN/products-and-services/self-service/Atlantis-Feature-Store
    
    path_read = f"{CREDENTIALS['paths']['scratch_root_personal']}/custom_churn"
    list_files = dbutils.fs.ls(path_read)
    if not list_files:   # guess that we didn't have anything written here yet...
        # load raw features from the feature store (this is an *advanced* topic that we'll review in hour 3!)
        df_churn_raw = churn_load_fs()
        df_custom_enriched = (df_custom_data
            .sample(0.1, seed=42)    # remember always start small to test!
            .join(df_churn_raw, df_custom_data['cloc'] == df_churn_raw['Acct_id'])
        )
        display(df_churn_raw.limit(10))   

        fn_log("Saving custom churn data...")
        df_churn_raw.write.format('delta').save(path_read)

    # perform normal read
    df_churn_raw = spark.read.format('delta').load(path_read)


# COMMAND ----------

# MAGIC %md
# MAGIC # Wrap-up
# MAGIC That's it for this section of exploring data, what did we learn?
# MAGIC 
# MAGIC * How you can upload a file to your own Azure blob storage
# MAGIC * Some example code for using the feature store
# MAGIC * Hoepfully a little experimentation with your own uploaded data (but that's on you ;)!
# MAGIC 
# MAGIC You can jump back to the workshop **02_predict_debiased** notebook to resume bias exploration and mitigation methods.
