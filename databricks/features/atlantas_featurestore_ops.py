# Databricks notebook source
# MAGIC %run ../utilities/settings

# COMMAND ----------

from featurestore import *
import os
os.environ['no_proxy'] = 'atlantis-feature-store.SERVICE_SITE'
client = Client('atlantis-feature-store.SERVICE_SITE:8080')

Token='Enter your PAT token'
client.auth.set_auth_token(Token)

project = client.projects.get('Neustar')
fs = project.feature_sets.get('Neustar Demographics')
data_reference = fs.retrieve()
df=data_reference.as_spark_frame(spark)
display(df)

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.functions import date_format, when, countDistinct
from pyspark.sql.functions import rand, max
import pyspark.sql.functions as F
import datetime as dt
