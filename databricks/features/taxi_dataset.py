# Databricks notebook source
# 
# This file contains taxi dataset utilities used for the 
#   2022 Machine Learning Workshop (part of the Software Symposium)!
#   https://FORWARD_SITE/mlworkshop2022 
#      OR https://INFO_SITE/cdo/events/internal-events/4354c5db-3d3d-4481-97c4-8ad8f12686f1
#
# You can (and should) change them for your own experiments, but they are uniquely defined
# here for constants that we will use together.

# COMMAND ----------

# MAGIC %run ../utilities/settings

# COMMAND ----------

import pyspark.sql.functions as F
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.window import Window

# COMMAND ----------

def taxi_plot_timeline(df, col_value='mean_total', vol_volume='volume', col_value_extra=None):
    # graph example from great plotting how-to site...
    #    https://www.python-graph-gallery.com/line-chart-dual-y-axis-with-matplotlib
    pdf_taxi_stats = (df
        .toPandas()
        .sort_values('date_trunc')
    )
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    COLOR_VOLUME = 'tab:blue'
    COLOR_PRICE = 'tab:orange'
    COLOR_PRICE_EXTRA = 'tab:olive'

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()
    ax1.bar(pdf_taxi_stats['date_trunc'], pdf_taxi_stats[vol_volume], color=COLOR_VOLUME, width=1.0, label='volume')
    ax2.plot(pdf_taxi_stats['date_trunc'], pdf_taxi_stats[col_value], color=COLOR_PRICE, lw=2, label='total price')
    if col_value_extra is not None:
        ax2.plot(pdf_taxi_stats['date_trunc'], pdf_taxi_stats[col_value_extra], color=COLOR_PRICE_EXTRA, lw=2, label='comparison price')
    # want to see some other weird stats? min/max are crazy (but we'll fix that later)
    # ax2.plot(pdf_taxi_stats['date_trunc'], pdf_taxi_stats['min_total'], color='violet', lw=2)
    # ax2.plot(pdf_taxi_stats['date_trunc'], pdf_taxi_stats['max_total'], color='firebrick', lw=2)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total Ride Volume", color=COLOR_VOLUME, fontsize=14)
    ax1.tick_params(axis="y", labelcolor=COLOR_VOLUME)

    ax2.set_ylabel("Average Total Fare ($)", color=COLOR_PRICE, fontsize=14)
    ax2.tick_params(axis="y", labelcolor=COLOR_PRICE)
    ax2.legend()

    fig.autofmt_xdate()
    fig.suptitle("NYC Taxi Volume and Average Fares", fontsize=20)


    # compute some overall stats
    fn_log(f"Date Range: {pdf_taxi_stats['date_trunc'].min()} - {pdf_taxi_stats['date_trunc'].max()} (total days {len(pdf_taxi_stats)})")
    fn_log(f"Total Rides: {pdf_taxi_stats[vol_volume].sum()}")
    fn_log(f"Avg Fare: {pdf_taxi_stats[col_value].mean()}")


# COMMAND ----------


def taxi_filter_raw(df_input):
    import datetime as dt
    # this is a helper function to filter out bad data by time range
    return (df_input
        .filter(F.col('pickup_datetime') >= F.lit(dt.datetime(year=2009, day=1, month=1)))  # after Jan 1, 2009
        .filter(F.col('pickup_datetime') <= F.lit(dt.datetime(year=2020, day=1, month=1)))  # before Jan 1, 2020
        .dropna(subset=['fare_amount'])  # drop any records that have empty/null fares
        .filter(F.col('fare_amount') >= F.lit(0.0))  # no negative-dollar fares
        .filter(F.col('fare_amount') <= F.lit(20000.0))  # no super-expensive fares
    )


# COMMAND ----------

def taxi_postproc_volumes(df_predict):
    # also write predicted, aggregated stats by the zone
    return (df_predict
        # do aggregate, but make temp copy for those in top category
        .withColumn('_pred_volume', F.when(F.col('is_top_predict')==F.lit(1), F.col('volume'))
                                   .otherwise(F.lit(0)))
        .withColumn('_pred_total', F.when(F.col('is_top_predict')==F.lit(1), F.col('total_amount'))
                                   .otherwise(F.lit(0)))
        .withColumn('_top_volume', F.when(F.col('is_top')==F.lit(1), F.col('volume'))
                                   .otherwise(F.lit(0)))
        .withColumn('_top_total', F.when(F.col('is_top')==F.lit(1), F.col('total_amount'))
                                   .otherwise(F.lit(0)))
        .groupBy('pickup_zone').agg(
            F.mean(F.col('total_amount')).alias('mean_total'),
            F.mean(F.col('_pred_total')).alias('mean'),
            F.sum(F.col('total_amount')).alias('sum_total'),
            F.sum(F.col('_pred_total')).alias('sum'),
            F.sum(F.col('volume')).alias('volume_total'),
            F.sum(F.col('_pred_volume')).alias('volume'),
        )
    )

# COMMAND ----------

def taxi_zone_demos(factor_limit=None, path_demos=None, compute_quantile=False):
    """Create a set of features from demographic factors that have been grouped by zone.
         Provide a specific factor for a specific set of features or None for all.
    """
    if path_demos is None:
        path_demos = CREDENTIALS['paths']['demographics_factors']
    df_demo_pivot_feat = (spark.read.format('delta').load(path_demos)
        .filter(F.col('factor').isNotNull())
    )
    if factor_limit is not None:
        df_demo_pivot_feat = df_demo_pivot_feat.filter(F.col('factor')==F.lit(factor_limit))

    # resume with feature pivot
    winFactor = Window.partitionBy('zone', 'factor')
    df_demo_pivot_feat = (df_demo_pivot_feat
        .withColumn("cnt_mean", F.mean(F.col('count')).over(winFactor))
        .withColumn("cnt_std", F.stddev(F.col('count')).over(winFactor))
        .withColumn("cnt_zscore", (F.col('count') - F.col('cnt_mean'))/F.col('cnt_std'))
        # use statistics to find quartiles in raw data
        # convert Z-score to quantile - https://mathbitsnotebook.com/Algebra2/Statistics/STstandardNormalDistribution.html
        .withColumn("cnt_quantile", F.when(F.col('cnt_zscore') >= (F.col('cnt_zscore') * F.lit(0.67448)), F.lit(1))
                                    .when(F.col('cnt_zscore') >= (F.col('cnt_zscore') * F.lit(0.0)), F.lit(2))
                                    .when(F.col('cnt_zscore') >= (F.col('cnt_zscore') * F.lit(-0.67448)), F.lit(3))
                                    .otherwise(F.lit(4)) )
        .withColumn('factor_val', F.regexp_replace(F.concat(F.col('factor'), F.lit('_'), 
                                    F.regexp_replace(F.col('value'), r"[^0-9a-zA-Z]+", "_")), r"[_]+", "_"))
    )
    # # partition to keep only samples in the quantile?
    # if quantile_min is not None:
    #     df_demo_pivot_feat = df_demo_pivot_feat.filter(F.col(cnt_quantile) <= F.lit(quantile_min))
    # if quantile_max is not None:
    #     df_demo_pivot_feat = df_demo_pivot_feat.filter(F.col(cnt_quantile) > F.lit(quantile_max))
    
    # finally, consolidate into pivot rable
    if compute_quantile:
        df_demo_pivot_feat = (df_demo_pivot_feat    
            .groupBy('zone').pivot('factor_val').agg(
                F.first('cnt_quantile').alias('cnt_quantile'),
            )
        )
    else:
        df_demo_pivot_feat = (df_demo_pivot_feat    
            .groupBy('zone').pivot('factor_val').agg(
                F.mean('cnt_zscore').alias('cnt_zscore'),
                #F.mean('count').alias('count'),
            )
        )
    return df_demo_pivot_feat.drop('null')



# COMMAND ----------

def taxi_zone_demo_disparities(factor_limit, path_demos=None, quantile_size=10):
    """Across zones and a demographics factor, allow the sub-classes of a factor to pick their
        zones of highest disparity.  Effectively, this should highlight zones with the biggest disparities
        across all zones so that we can identify a "protected" class.
    """
    winFactor = Window.partitionBy('zone')
    winDisparity = Window.partitionBy('factor').orderBy(F.col('cnt_zscore').desc())
    winZone = Window.partitionBy('zone').orderBy(F.col('cnt_zscore').desc())

    if path_demos is None:
        path_demos = CREDENTIALS['paths']['demographics_factors']
    df_disparity = (spark.read.format('delta').load(path_demos)
        .filter(F.col('factor').isNotNull())
        .filter(F.col('factor')==F.lit(factor_limit))
        .withColumn("cnt_mean", F.mean(F.col('count')).over(winFactor))
        .withColumn("cnt_std", F.stddev(F.col('count')).over(winFactor))
        .withColumn("cnt_zscore", (F.col('count') - F.col('cnt_mean'))/F.col('cnt_std'))
        .withColumn("cnt_quantile", F.ntile(quantile_size).over(winZone))
        # convert Z-score to quantile - https://mathbitsnotebook.com/Algebra2/Statistics/STstandardNormalDistribution.html
        # .withColumn("cnt_quantile", F.when(F.col('cnt_zscore') >= (F.col('cnt_zscore') * F.lit(0.67448)), F.lit(1))
        #                             .when(F.col('cnt_zscore') >= (F.col('cnt_zscore') * F.lit(0.0)), F.lit(2))
        #                             .when(F.col('cnt_zscore') >= (F.col('cnt_zscore') * F.lit(-0.67448)), F.lit(3))
        #                             .otherwise(F.lit(4)) )
        .withColumn("overall_quantile", F.ntile(quantile_size).over(winDisparity))
    )
    return df_disparity

