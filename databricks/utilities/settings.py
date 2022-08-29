# Databricks notebook source
# 
# This file contains settings utilities used for the 
#   2022 Machine Learning Workshop (part of the Software Symposium)!
#   https://FORWARD_SITE/mlworkshop2022 
#      OR https://INFO_SITE/cdo/events/internal-events/4354c5db-3d3d-4481-97c4-8ad8f12686f1
#
# You can (and should) change them for your own experiments, but they are uniquely defined
# here for constants that we will use together.


# COMMAND ----------

# this code chunk defines a custom logging function
from os import environ
IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in environ
try:
    if logger is None:
        pass
except Exception as e:
    logger = None

if logger is None:
    import logging
    logger = logging.getLogger(__name__)
    if IS_DATABRICKS:
        from sys import stderr, stdout
        # writing to stdout
        handler = logging.StreamHandler(stdout)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

        # See this thread for logging in databricks - https://stackoverflow.com/a/34683626; https://github.com/mspnp/spark-monitoring/issues/50#issuecomment-533778984
        log4jLogger = spark._jvm.org.apache.log4j
#     logger = log4jLogger.Logger.getLogger(__name__)  # --- disabled for whitelisting in 8.3 ML 

def fn_log(str_print):   # place holder for logging
    logger.info(str_print)
    print(str_print)

# COMMAND ----------

# This is utility code to detect the current user.  It works on the AIaaS cluster to detecct the user ATTID
# which is often the defauly location for new notebooks, repos, etc.

# template for reference, please add specific values value
try:
    if CREDENTIALS is not None:
        print("Aborting re-definition of credentials...")
except Exception as e:
    CREDENTIALS = {'credentials': {}, 'paths': {}, 'constants': {} }
    fn_log(f"Credentials not detected, creating a new one ({e})...")

    # also truncate GIT errors (seen as databricks error with mlflow)
    environ['GIT_PYTHON_REFRESH'] = 'quiet'
    
    
    # https://stackoverflow.com/a/64039504
    # %sql
    # set spark.databricks.userInfoFunctions.enabled = true;
    spark.conf.set("spark.databricks.userInfoFunctions.enabled", "true")
    USER_PATH = spark.sql('select current_user() as user;').collect()[0]['user']
    CREDENTIALS['credentials']['ATTID'] = USER_PATH.split('@')[0]
    fn_log(f"Detected User ID: {CREDENTIALS['credentials']['ATTID']}")

    # tokens for UPSTART + FEATURE STORE
    # this is an example of using secrets; it's an "advanced" topic (https://wiki.SERVICE_SITE/x/2gpiWQ), 
    # but your cluster admin should probably do this for you.  don't have one? just here for fun?
    # NO PROBLEM!  Just update the two settings below and delete the comments.
    # JUST REMEMBER... never check in or share your code with these credentials set!
    try:
        CREDENTIALS['credentials']['UPSTART'] = dbutils.secrets.get(
            scope=f"dsair-{CREDENTIALS['credentials']['ATTID']}-scope", 
            key=f"{CREDENTIALS['credentials']['ATTID']}-upstart")
        CREDENTIALS['credentials']['ATLANTIS'] = dbutils.secrets.get(
            scope=f"dsair-{CREDENTIALS['credentials']['ATTID']}-scope", 
            key=f"{CREDENTIALS['credentials']['ATTID']}-atlantis")
    except Exception as e:
        CREDENTIALS['credentials']['UPSTART'] = "your own password"
        CREDENTIALS['credentials']['ATLANTIS'] = "your own token"

        # you can hide these complaints when you add your PW here
        fn_log(f"Could not load UPSTART credentials for ATTID {CREDENTIALS['credentials']['ATTID']}...")
        fn_log(f"This will prevent direct reads from Snowflake sources.")
        fn_log(f"Not what you expected? Check out `utilities/settings.py` for remediations.")
        fn_log(f"Here's the actual error: {e}.")

    # tokens for DEEP
    # CREDENTIALS['credentials']['DEEP'] = dbutils.secrets.get(scope=f"dsair-{CREDENTIALS['credentials']['ATTID']}-scope", key='deep_token')
    # specific locations for this project (in a user's directory)
    # tokens for write locations
    CREDENTIALS['paths'] = {'notebooks': f'/Users/{USER_PATH}', 'repos': f'/Repos/{USER_PATH}', 
                            'user': f"/user/{CREDENTIALS['credentials']['ATTID']}" }

def is_workshop_admin():
    return CREDENTIALS['credentials']['ATTID'] in ['ez2685']

# COMMAND ----------

def quiet_delete(path_storage):
    try:
        dbutils.fs.rm(path_storage, True)   # recursively delete what was there before
    except Excpetion as e:
        fn_log(f"Error clearing parititon '{path_storage}', maybe it didn't exist...")


# COMMAND ----------

import datetime as dt

# okay to always update these lines because they cost no compute

# update for the local paths
CREDENTIALS['paths'].update({
  'experiments': f"{CREDENTIALS['paths']['user']}/experiments",
})


# the last command assumes you make a scratch directory that you own (or can write to) with your ATTID
# for instructions on how to create your own scratch, head to notebook `1b_DATA_WRITE_EXAMPLES`
CREDENTIALS['paths'].update({
    'scratch_root_personal': f"abfss://{CREDENTIALS['credentials']['ATTID']}@STORAGE"
})

if dt.datetime.now() > dt.datetime(month=10, year=2022, day=21):     # the shared scratch will be disabled after Oct 21
    CREDENTIALS['paths'].update({'scratch_root': CREDENTIALS['paths']['scratch_root_personal']})

# No time to make your own scratch? no problem, you can use the temp container created for the workshop.
# NOTE: this container will be deactivated within a week or so of the workshop!
else:
    CREDENTIALS['paths'].update({'scratch_root': f"abfss://mlworkshop2022-writeable@STORAGE/{CREDENTIALS['credentials']['ATTID']}"})

# define constants and reusable strings here
CREDENTIALS['paths'].update({
    'databricks_dataset': "abfss://mlworkshop2022@dsairgeneraleastus2sa.STORAGE", # "abfss://mlworkshop2022@STORAGE",
    'geometry_base': "abfss://mlworkshop2022@dsairgeneraleastus2sa.STORAGE/tl_2021_us_zcta520/geometry", 
})
CREDENTIALS['paths'].update({
    # datasets that we may consider joining
    'geometry_zip': f"{CREDENTIALS['paths']['geometry_base']}/ztca",
    #    zip code data from census - https://www2.census.gov/geo/tiger/TIGER2021/ZCTA520/
    'geometry_dma': f"{CREDENTIALS['paths']['geometry_base']}/dma",
    #    DMA shape data from neilsen archive - https://github.com/simzou/nielsen-dma
    'geometry_state': f"{CREDENTIALS['paths']['geometry_base']}/state",
    #    state shape data from census - https://www2.census.gov/geo/tiger/TIGER2021/STATE/
    'geometry_county': f"{CREDENTIALS['paths']['geometry_base']}/county",
    #    dataset for county - https://www2.census.gov/geo/tiger/TIGER2021/COUNTY/
    'geometry_nyctaxi': f"{CREDENTIALS['paths']['geometry_base']}/nyctaxi_zones",
    #    dataset for nyc boroughs - https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/d3c5-ddgc
    'demographics_raw': f"{CREDENTIALS['paths']['databricks_dataset']}/demographics",
    #    dataset of demographics from internal subset from AT&T vendor

    # datasets that you may have created
    'nyctaxi_raw': f"{CREDENTIALS['paths']['databricks_dataset']}/nyctaxi/tables/nyctaxi_yellow",  # raw dataset
    'nyctaxi_stats': f"{CREDENTIALS['paths']['databricks_dataset']}/experiment/nyctaxi_stats",   # overall stats
    'nyctaxi_stats_cleaned': f"{CREDENTIALS['paths']['databricks_dataset']}/experiment/nyctaxi_stats_cleaned",   # refined data stat
    'nyctaxi_h3': f"{CREDENTIALS['paths']['databricks_dataset']}/experiment/nyctaxi_h3",   # encoded with h3
    'nyctaxi_h3_sampled': f"{CREDENTIALS['paths']['databricks_dataset']}/experiment/nyctaxi_h3_sampled",  # subsampled with h3
    'nyctaxi_h3_zips': f"{CREDENTIALS['paths']['databricks_dataset']}/experiment/nyctaxi_h3_zips",   # zip stat data
    'nyctaxi_h3_zones': f"{CREDENTIALS['paths']['databricks_dataset']}/experiment/nyctaxi_h3_zones",   # zone stat data
    'demographics_factors': f"{CREDENTIALS['paths']['databricks_dataset']}/experiment/demographics_factors",   # groups by various factors
    'nyctaxi_geo_sampled': f"{CREDENTIALS['paths']['databricks_dataset']}/experiment/nyctaxi_geo_sampled",  # geo+features write
})

# need to have time constants restructed?
dt_now_month = dt.datetime(year=dt.datetime.now().year, month=dt.datetime.now().month, day=1)
CREDENTIALS['constants'].update({
    'MLFLOW_EXPERIMENT': "MLWorkshop2022",
    'EXPERIENCED_MODE': False,   # change this flag to run some processing in 'experienced' mode
    'WORKSHOP_ADMIN_MODE': CREDENTIALS['credentials']['ATTID'] in ['ez2685'],
    'DATA_SUBSAMPLE_RATIO': 0.03,   # from our raw data, grab only 3% for quicker demos in this workshop!

    # our spatial mapping uses h3 cells (resolution 11?) - 
    # see https://towardsdatascience.com/uber-h3-for-data-analysis-with-python-1e54acdcc908
    #    according to h3 resolution - https://h3geo.org/docs/core-library/restable/#average-area-in-m2
    #    resolution 11 is about 23138.565 square feet
    #    resolution 10 (roughly a city block) is about 161970 square feet
    'RESOLUTION_H3': 11,  
    
    # this may not work for sure, but let's try to format an Azure Portal for above...
    'SCRATCH_URL': f"https://PORTAL/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2F81b4ec93-f52f-4194-9ad9-57e636bcd0b6%2FresourceGroups%2Fblackbird-prod-storage-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fblackbirdproddatastore/path/mlworkshop2022/etag/%220x8DA82A21AA0D1FA%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None",
    'TIME_SAMPLING_LIMITS': {
        "last_year": {'dt_start': dt.datetime(year=dt_now_month.year-1, month=1, day=1), 
                      'dt_end': dt.datetime(year=dt_now_month.year-1, month=12, day=31) },
        "last_month": {'dt_start': dt.datetime(year=dt_now_month.year, month=(dt_now_month - dt.timedelta(days=1)).month, day=1), 
                      'dt_end': dt_now_month },
    },

})




# COMMAND ----------


