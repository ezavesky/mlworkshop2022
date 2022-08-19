# Databricks notebook source
# 
# Databricks notebook source
# This file contains constants used for the workshop!
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

    # tokens for UPSTART
    try:
        CREDENTIALS['credentials']['UPSTART'] = dbutils.secrets.get(
            scope=f"dsair-{CREDENTIALS['credentials']['ATTID']}-scope", 
            key=f"{CREDENTIALS['credentials']['ATTID']}-upstart")
    except Exception as e:
        fn_log(f"Could not load UPSTART credentials for ATTID {CREDENTIALS['credentials']['ATTID']}...")
        fn_log(f"This will prevent direct reads from terradata (eCDW) and Snowflake sources.")
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

# okay to always update these lines because they cost no compute

# update for the local paths
CREDENTIALS['paths'].update({
  'experiments': f"{CREDENTIALS['paths']['user']}/experiments",
})


# the last command assumes you make a scratch directory that you own (or can write to) with your ATTID
# for instructions on how to create your own scratch, head to notebook `1b_DATA_WRITE_EXAMPLES`
if dt.datetime.now() > dt.datetime(month=10, year=2022, day=21):     # the shared scratch will be disabled after Oct 21
    CREDENTIALS['paths'].update({'scratch_root', f"abfss://{USER_ID}@STORAGE"}

# No time to make your own scratch? no problem, you can use the temp container created for the workshop.
# NOTE: this container will be deactivated within a week or so of the workshop!
else:
    CREDENTIALS['paths'].update({'scratch_root', f"abfss://mlworkshop2022-writeable@STORAGE/{USER_ID}"})

# define constants and reusable strings here
CREDENTIALS['paths'].update({
    'databricks_dataset': "abfss://mlworkshop2022@STORAGE",
    # map mapping for enc/dec
    'msp_records': "abfss://msp@datalakeeastus2prd.STORAGE/msp_nation_aggregate_final",
    'msp_mapping': "abfss://msp-mapping@datalakeeastus2prd.STORAGE/msp_nation_aggregate_final_dlmapping",
    'scamp_data': "abfss://scamp-att@datalakeeastus2prd.STORAGE/scamp_awsd_att",
})
CREDENTIALS['paths'].update({
    # we may delete this set in future? seems to be similar to 'present' and has team-specific encoding ? 
})

# need to have time constants restructed?
import datetime as dt
dt_now_month = dt.datetime(year=dt.datetime.now().year, month=dt.datetime.now().month, day=1)
CREDENTIALS['constants'].update({
    'MLFLOW_EXPERIMENT': "MLWorkshop2022",
    # this may not work for sure, but let's try to format an Azure Portal for above...
    'SCRATCH_URL': f"https://PORTAL/#blade/Microsoft_Azure_Storage/ContainerMenuBlade/overview/storageAccountId/%2Fsubscriptions%2F81b4ec93-f52f-4194-9ad9-57e636bcd0b6%2FresourceGroups%2Fblackbird-prod-storage-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fblackbirdproddatastore/path/{USER_ID}/etag/%220x8D9766DE75EA338%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride//defaultId//publicAccessVal/None",
    'TIME_SAMPLING_LIMITS': {
        "last_year": {'dt_start': dt.datetime(year=dt_now_month.year-1, month=1, day=1), 
                      'dt_end': dt.datetime(year=dt_now_month.year-1, month=12, day=31) },
        "last_month": {'dt_start': dt.datetime(year=dt_now_month.year, month=(dt_now_month - dt.timedelta(days=1)).month, day=1), 
                      'dt_end': dt_now_month },
    },

})




# COMMAND ----------


