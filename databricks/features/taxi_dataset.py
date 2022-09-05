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

