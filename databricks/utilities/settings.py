# Databricks notebook source
# 
# The following code provides project-level constants as
# part of the ML Workshop 2022 project
#  


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


# COMMAND ----------

# legaccy function
def get_creds():
    """function to return current user"""
    # spark.conf.set("spark.databricks.userInfoFunctions.enabled", "true")
    # USER_PATH = spark.sql('select current_user() as user;').collect()[0]['user']
    # ATTID = USER_PATH.split('@')[0]
    # UPSTART_PWD = dbutils.secrets.get(scope=f'dsair-{ATTID}-scope', key=f'{ATTID}-upstart')
    print("... Legacy function `get_creds` please update your code to use the project constants file!")
    return CREDENTIALS['credentials']['ATTID'], CREDENTIALS['credentials']['UPSTART']


# COMMAND ----------

# okay to always update these lines because they cost no compute

# update for the local paths
CREDENTIALS['paths'].update({
  'experiments': f"{CREDENTIALS['paths']['user']}/experiments",
})

# define constants and reusable strings here
CREDENTIALS['paths'].update({
    'databricks_dataset': "abfss://dsair-geo-loc-fencing@dsairgeneraleastus2sa.STORAGE",
    # map mapping for enc/dec
    'msp_records': "abfss://msp@datalakeeastus2prd.STORAGE/msp_nation_aggregate_final",
    'msp_mapping': "abfss://msp-mapping@datalakeeastus2prd.STORAGE/msp_nation_aggregate_final_dlmapping",
    'scamp_data': "abfss://scamp-att@datalakeeastus2prd.STORAGE/scamp_awsd_att",
    'scamp_sms': "abfss://scamp-att@datalakeeastus2prd.STORAGE/scamp_smsd_att",
    'clnph': "abfss://dsair-geo-loc-fencing@dsairgeneraleastus2sa.STORAGE/closenuph",
    'cssng': "abfss://cssng@datalakeeastus2prd.STORAGE/ss_site_master",
})
CREDENTIALS['paths'].update({
    'clnph_bronze': f"{CREDENTIALS['paths']['databricks_dataset']}/closenuph",
    'locstore_bronze': f"{CREDENTIALS['paths']['databricks_dataset']}/locstore",
    #'msdin_bronze': f"{CREDENTIALS['paths']['databricks_dataset']}/msisdn_dec",  # typo, can be deleted? (TO BE DELETED....)
    #'msisdn_decode': f"{CREDENTIALS['paths']['databricks_dataset']}/msisdn_dec",   # (TO BE DELETED....)
    'events': f"{CREDENTIALS['paths']['databricks_dataset']}/events",  # set of known events
    #'present_bronze': f"{CREDENTIALS['paths']['databricks_dataset']}/present",   # TO BE DELETED....
    'fanshed_bronze': f"{CREDENTIALS['paths']['databricks_dataset']}/fanbehavior",
    # paths for pipeline (started 4/28)
    'present_parsed': f"{CREDENTIALS['paths']['databricks_dataset']}/present_parsed",  # parsed input event data
    #'present_cohorts': f"{CREDENTIALS['paths']['databricks_dataset']}/present_cohorts",  # filtered imsis  (TO BE DELETED 8/7, is it still used?)
    'present_optin_imei': f"{CREDENTIALS['paths']['databricks_dataset']}/present_optin",  # filtered + opt-in + imei
    'location_optin_msisdn': f"{CREDENTIALS['paths']['databricks_dataset']}/location_optin",  # filtered + opt-in + location
    'geofence_laccids': f"{CREDENTIALS['paths']['databricks_dataset']}/geofence_laccids",   # CSV for laccids (to be update?)

    # fixed or other individual data stores (used sparingly)
    'project_scratch': f"{CREDENTIALS['paths']['databricks_dataset']}/scratch/{CREDENTIALS['credentials']['ATTID']}",    
    'geojson_combined': f"{CREDENTIALS['paths']['databricks_dataset']}/geojson/combined",
    # we may delete this set in future? seems to be similar to 'present' and has team-specific encoding ? 
    'present_fan_bronze': f"{CREDENTIALS['paths']['databricks_dataset']}/TBLfans",    #can we delete this?
    'fan_pictures': f"{CREDENTIALS['paths']['databricks_dataset']}/fanbehavior/pictures",
})

# need to have time constants restructed?
import datetime as dt
dt_now_month = dt.datetime(year=dt.datetime.now().year, month=dt.datetime.now().month, day=1)
CREDENTIALS['constants'].update({
    'TIME_SAMPLING_LIMITS': {
        "last_year": {'dt_start': dt.datetime(year=dt_now_month.year-1, month=1, day=1), 
                      'dt_end': dt.datetime(year=dt_now_month.year-1, month=12, day=31) },
        "last_month": {'dt_start': dt.datetime(year=dt_now_month.year, month=(dt_now_month - dt.timedelta(days=1)).month, day=1), 
                      'dt_end': dt_now_month },
    },
    'TIME_LOCATION_SHED_WINDOW': dt.timedelta(days=3),
    'TIME_BROWSER_WINDOW': dt.timedelta(days=7),
    'TIME_ATTEND_MIN_THRESHOLD': 60,
    'TIME_DURATION_MIN_THRESHOLD': 390,
})



# COMMAND ----------


