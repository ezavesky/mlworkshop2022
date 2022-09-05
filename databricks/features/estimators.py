# Databricks notebook source
# 
# This file contains modeling helper functions for the 
#   2022 Machine Learning Workshop (part of the Software Symposium)!
#   https://FORWARD_SITE/mlworkshop2022 
#      OR https://INFO_SITE/cdo/events/internal-events/4354c5db-3d3d-4481-97c4-8ad8f12686f1
#
# You can (and should) change them for your own experiments, but they are uniquely defined
# here for constants that we will use together.


# COMMAND ----------

from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark import keyword_only
import numpy as np

# COMMAND ----------

class HasTunedThreshold(Params):
    tunedThreshold = Param(Params._dummy(),
            "tunedThreshold", "tunedThreshold",
            typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasTunedThreshold, self).__init__()

    def setTunedThreshold(self, value):
        return self._set(tunedThreshold=value)

    def getTunedThreshold(self):
        return self.getOrDefault(self.tunedThreshold)
    
    
class TunedThreshold(Estimator, HasLabelCol, HasPredictionCol, HasInputCol,
                      DefaultParamsReadable, DefaultParamsWritable):

    @keyword_only
    def __init__(self, labelCol=None, inputCol=None, predictionCol=None):
        super(TunedThreshold, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    # Required in Spark >= 3.0
    def setLabelCol(self, value):
        """
        Sets the value of :py:attr:`labelCol`.
        """
        return self._set(labelCol=value)

    # Required in Spark >= 3.0
    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    # Required in Spark >= 3.0
    def setPredictionCol(self, value):
        """
        Sets the value of :py:attr:`predictionCol`.
        """
        return self._set(predictionCol=value)

    @keyword_only
    def setParams(self, inputCol=None, labelCol=None, predictionCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    # https://stackoverflow.com/a/46077238
    @staticmethod
    def _compute_turning_threshold(event_rate, y_prob):
        threshold = np.percentile(y_prob, 100 - event_rate)
        # y_pred = [1 if x >= threshold else 0 for x in y_prob[:, 1]]
        return threshold

    # https://stackoverflow.com/a/46077238
    @staticmethod
    def compute_turning_threshold(y_truth, y_prob):
        event_rate = sum(y_truth) / len(y_prob) * 100
        return _compute_turning_threshold(event_rate, y_prob)

    def _fit(self, dataset):
        col_prob = self.getInputCol()    # the probability column
        col_label = self.getLabelCol()    # true label
        col_pred = self.getPrecitionCol()    # final prediction
        
        # see `compute_turning_threshold`
        event_rate = dataset.agg((F.sum(col_label) / F.count(col_label) * F.lit(100)).alias('thresh')).collect()[0]['thresh']
        threshold = TunedThreshold._compute_turning_threshold(event_rate, dataset.select(col_label).toPandas())
        return NormalDeviationModel(
            inputCol=col_prob, tunedThreshold=threshold, predictionCol=col_pred)


class TunedThresholdModel(Model, HasPredictionCol, HasInputCol, HasTunedThreshold, 
                          DefaultParamsReadable, DefaultParamsWritable):

    @keyword_only
    def __init__(self, inputCol=None, predictionCol=None, tunedThreshold=None):
        super(TunedThresholdModel, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)  

    @keyword_only
    def setParams(self, inputCol=None, predictionCol=None, tunedThreshold=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)           

    def _transform(self, dataset):
        udf_last = F.udf(lambda v:float(v[-1]), T.FloatType())
        col_prob = self.getInputCol()
        col_pred = self.getPredictionCol()
        threshold = self.getTunedThreshold()
        col_temp = f"_flat_{col_pred}"
        return (dataset
            .drop(col_pred)
            .withColumn(col_temp, udf_last(F.col(col_pred)))
            .withColumn(col_pred, F.when(F.col(col_temp) >= F.lit(_threshold), F.lit(1))
                                  .otherwise(F.lit(0)))
            .drop(col_temp)
        )

