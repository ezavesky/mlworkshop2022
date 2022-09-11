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

# MAGIC %run ../utilities/mlflow

# COMMAND ----------

# MAGIC %run ./estimators

# COMMAND ----------

import pyspark.sql.types as T
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.feature import StringIndexer, IndexToString, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import functions as F
import numpy as np

# COMMAND ----------

def modeling_gridsearch(df_train, col_label, num_folds=3, parallelism=1):
    """General search across a known grid using training data and a label target
            general_gridsearch('run', df_train, col_label)
    """
    import mlflow
    from hyperopt import fmin, tpe, STATUS_OK, Trials, SparkTrials
    from hyperopt import hp

    list_category = [x[0] for x in df_train.dtypes if x[1]=='string']
    list_category_int = [n+"_INT" for n in list_category]
    list_numeric = list(set(list_features) - set(list_category))
    fn_log(f"[learning_gridsearch] Categorical [{list_category}]; Numeric {list_numeric}")

    ### --- so that we can do distributed training, there are several helper functions inlined below ---
    #   if you don't understand why we did this, it's because that's what hyperopt (databricks suggested training)
    #   requires as its way to parallelize training.  essentially, we have the OPTION to do k-fold training
    #   but honestly you may not need to    
    
    def _get_classifier_rf(max_bins=1000, params={}):
        # in case there is some overflow of categorical
        rf = RandomForestClassifier(featuresCol='features', labelCol=col_label, 
                                            predictionCol=f"{col_label}_predict", probabilityCol=f"{col_label}_prob_vect",
                                            rawPredictionCol=f"{col_label}_prob_raw", maxBins=max_bins, **params)
        return rf

    def _get_untrained_pipeline(cf=None):
        category_index = StringIndexer(inputCols=list_category, outputCols=list_category_int, 
                                        stringOrderType="frequencyDesc", handleInvalid ='keep')
        vec_assemble = VectorAssembler(inputCols=list_category_int+list_numeric, outputCol="features")
        if cf is None:
            cf = _get_classifier_rf()  
        norm_thresh = TunedThreshold(inputCol=f"{col_label}_prob_vect", predictionCol=f"{col_label}_predict", 
                                     outputCol=f"{col_label}_probability", labelCol=col_label)
        pipeline_full = Pipeline(stages=[category_index, vec_assemble, cf, norm_thresh])
        return pipeline_full

     # helper for RF
    def _fn_train_rf(params, metric_name='areaUnderROC'):
        # create a grid with our hyperparameters
        cf = _get_classifier_rf()
        pipe = _get_untrained_pipeline(cf)
        
        param_grid = ParamGridBuilder()
        for param_name in params:
            param_grid.addGrid(getattr(cf, param_name), [params[param_name]])
        grid = param_grid.build()
        fn_log(f"[_fn_train_rf] OBJECTIVE PARAMS (RF): {params}, Metric: {metric_name}")

        # cross validate the set of hyperparameters        
        evaluator = BinaryClassificationEvaluator(rawPredictionCol=col_label+'_prob_raw', 
                                                      labelCol=col_label, 
                                                      metricName=metric_name)
        cv = CrossValidator(estimator=pipe, evaluator=evaluator,
                            estimatorParamMaps=grid,
                            numFolds=num_folds, parallelism=parallelism, seed=42)
        cvModel = cv.fit(df_train)
        # get our average primary metric across all three folds (make negative because we want to minimize)
        retVal = sum(cvModel.avgMetrics)/len(cvModel.avgMetrics)
        # if evaluator.isLargerBetter():
        retVal = -retVal    # most metrics are better when larger, but we're MINIMIZING, so negate
        return {"loss": retVal, "params": params, "status": STATUS_OK}

    # search space for random forest
    search_space = {
      "numTrees": hp.choice("numTrees", [20, 50, 100, 200]),  # a few options, with uniform sampling
    }
    search_default = {'numTrees': 100}
    num_evals = 1
    
    # NOTE: can't specify parallelism for 'sparktrials' --- boo, this slows things way down
    # py4j.security.Py4JSecurityException: Method public int org.apache.spark.SparkContext.maxNumConcurrentTasks() is not whitelisted on class class org.apache.spark.SparkContext
    trials = Trials()   # only works with one object!

    ## STEP 1: search for best model using grid search if we have more than one
    # fold set.  if not, just use the default parameters (faster for debugging)
    if num_folds > 1:
        best_hyperparam = fmin(fn=_fn_train_rf,
                             space=search_space,
                             algo=tpe.suggest, 
                             max_evals=num_evals,
                             trials=trials,
                             rstate=np.random.RandomState(42)
                            )

        # the "trials" object has all the goodies this time!
        #   https://github.com/hyperopt/hyperopt/wiki/FMin#13-the-trials-object
        fn_log(f"[learning_gridsearch] MLflow Trials: {trials.trials}")
        fn_log(f"[learning_gridsearch] MLflow hyperparam (may be invalid): {best_hyperparam}")
        best_hyperparam = None
        best_loss = 1e10
        for g in trials.trials:
            if best_hyperparam is None or best_loss > g['result']['loss']:
                best_hyperparam = g['result']['params']
                best_loss = g['result']['loss']
            #for p in g['misc']['vals']:
            #    # mlflow.log_param(p, g['misc']['vals'][p][0])
            #    logger.info(p, g['misc']['vals'][p][0])
            fn_log(g['result'])
    else:
        best_hyperparam = search_default

    # okay, let's apply our best params and train the pipeline!
    fn_log(f"[learning_gridsearch] MLflow best hyperparam (may be invalid): {best_hyperparam}")
    cf = _get_classifier_rf(params=best_hyperparam)
    pipe = _get_untrained_pipeline(cf)
    return pipe, best_hyperparam



# COMMAND ----------

def modeling_train(df_train, df_test, col_label, model_name, pipeline, dict_param_extra,
                   list_inputs=None, name_experiment=CREDENTIALS['constants']['EXPERIMENT_NAME']):
    import datetime as dt
    import os  # for removing the temp file
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay, classification_report
    from matplotlib import pyplot as plt
    from mlflow.models.signature import infer_signature
    from functools import partial
    from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, accuracy_score

    # create experiment for mlflow to track!
    databricks_mlflow_create(name_experiment)
    dt_begin = dt.datetime.now()
    run_name = f"{model_name}_{dt_begin.strftime('%Y%m%d-%H%M%S')}"  # if we had some other models, we could consider them
    udf_last = F.udf(lambda v:float(v[-1]), T.FloatType())

    with mlflow.start_run(run_name=run_name) as run:
        # immediately run training and eval against test set
        clf = pipeline.fit(df_train)
        df_train_result = clf.transform(df_train)
        list_outputs = [f"{col_label}_predict", f"{col_label}_probability"]
        pdf_train = df_train_result.select(col_label, *list_outputs).toPandas()

        # you don't have to, but we've found that the natural calibration for 
        #   a threshold doesn't always end up at 0.5; so this helper detects that
        #   threshold on the training data and uses that to recompute the final prediction
        class_pos = 1
        # re-run on testing data
        df_pred = clf.transform(df_test)
        pdf_pred = df_pred.select(col_label, *list_outputs).toPandas()

        y_test = pdf_pred[col_label]
        test_pred = pdf_pred[f"{col_label}_predict"]
        test_score = pdf_pred[f"{col_label}_probability"]
        
        ## STEP 2: display some performance figures, like a confusion matrix and precision/recall curves
        # test_pred = clf.predict_proba(X_test)[:,1] > score_thresh
        cm = confusion_matrix(y_test, test_pred, normalize='all')
        test_auc = roc_auc_score(y_test, test_score)

        cm_fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        cm_display = ConfusionMatrixDisplay(cm).plot(ax=ax1)
        ax1.set_title(f"Total Samples: {len(pdf_pred)}")        

        # y_score = clf.predict_proba(X_test)[:,1]
        # y_score = y_pred

        fpr, tpr, _ = roc_curve(y_test, test_score, pos_label=class_pos)
        roc_fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.grid(True)
        ax1.plot([0, 1], [0, 1], linewidth=1, linestyle=":")
        ax1.set_title(f"AUC: {round(test_auc, 3)}")
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax1)

        prec, recall, _ = precision_recall_curve(y_test, test_score, pos_label=class_pos)
        pr_fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.grid(True)
        pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot(ax=ax1)
        n_pos_test = len(y_test[y_test > 0])
        n_neg_test = len(y_test[y_test < 1])
        ax1.set_title(f"PR Curve: (TN: {n_neg_test}, TP: {n_pos_test})")

        ## STEP 3: write out a text-like performance table to a text file
        dict_report = classification_report(y_test, test_pred, output_dict=True)

        # write some various metrics, params, etc.
        # mlflow.set_tag("s.release", "1.2.3")
        mlflow.set_tag("date", dt_begin.strftime('%Y%m%d'))
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("production", 0)

        # parameters that change the experiment/
        mlflow.log_param("model_name", model_name)
        n_pos = df_train.filter(F.col(col_label) > F.lit(0)).count()
        n_neg = df_train.filter(F.col(col_label) < F.lit(1)).count()
        mlflow.log_param("samples_pos", n_pos)
        mlflow.log_param("samples_neg", n_neg)
        mlflow.log_param("samples_ratio", float(n_neg)/n_pos)
        # mlflow.log_param("dt_earliest", dt_earliest.strftime('%Y%m%d-%H%M%S'))
        # mlflow.log_param("cols_eval", col_evaluation)
        if dict_param_extra is not None:  # logging extra parameters
            for k in dict_param_extra:
                mlflow.log_param(k, dict_param_extra[k])                

        # metrics that determine performance (and are searchable later!?)
        mlflow.log_metric("accuracy", dict_report["accuracy"])
        mlflow.log_metric("auc", test_auc)
        mlflow.log_metric("samples", n_pos+n_neg)

        # random text that might be interesting for a log
        #mlflow.log_text(f"{dt.datetime.now()}: this would be a complete log to save...", 'log_timestamp.txt')

        # plot a figure and log it to mlflow
        mlflow.log_figure(cm_fig, "model_confusion.png")
        mlflow.log_figure(roc_fig, "model_roc.png")
        mlflow.log_figure(pr_fig, "model_precision_recall.png")

        # close all of the figures in this iteration
        plt.close('all')

        # NOTE: use this sparingly = write a report to be associated with your new model...
        #mlflow.log_artifact(f.name, 'report')
        #os.unlink(f.name)
        #mlflow.log_dict(dict_report, 'report_dict.json')

        # and the model artifact
        if list_inputs is not None:
            df_input_sig = df_test.select(list_inputs).limit(10)
            df_output_sig = df_pred.select(list_outputs).limit(10)
            signature = infer_signature(df_input_sig, df_output_sig)
            # add an input examlpe
            input_example = df_input_sig.limit(1).toPandas()
            mlflow.spark.log_model(clf, model_name, input_example=input_example, signature=signature)
        else:
            mlflow.spark.log_model(clf, model_name)

        # reload your experiment view, we just wrote to a new run...
        fn_log(f"New run logged {run.info}...")
        
    return run_name, df_pred


# COMMAND ----------

# helper to predict on specific dataframe
def model_lookup(name_model=None, name_experiment=CREDENTIALS['constants']['EXPERIMENT_NAME']):
#     fn_log("[decisionmaker_predict] Broadcasting weights...")
    # first, look for models marked as proudciton
    tags_search = {'tags.production': 1}
    if name_model is not None:  # provide a specific product?
        tags_search['tags.model_name'] = name_model
    experiment_pdf = databricks_mlflow_load(name_experiment, tag_dict=tags_search)
    if not len(experiment_pdf):   # fallback to mon-production
        tags_search['tags.production'] = 0
        experiment_pdf = databricks_mlflow_load(name_experiment, tag_dict=tags_search)
    if not len(experiment_pdf):
        fn_log(f"[model_lookup]: Failed to retrieve valid model, aborting.")
        return None
    # filter to one object
    fn_log(f"[model_lookup]: Retrieved {len(experiment_pdf)} prior matches.")
    experiment_obj = experiment_pdf.iloc[0]
        
    # logged_model = 'dbfs:/Shared/DSAIR-projects/contacts_refinement/decisionmaker/b97e88eb7b204082a8a64edde8b48f7a/artifacts/lightgbm'
    # logged_model = 'runs:/ee746d2d1d644acd898c8e3bf2013f4c/lightgbm'    # live embedding
    # logged_model = 'runs:/58c8a6535492431b9a5c9bace8ff2471/lightgbm'    # cached embedding
    logged_model = f"runs:/{experiment_obj['run_id']}/{experiment_obj['params.model_name']}"
    fn_log(f"[model_lookup]: Retrieved valid model, run: {experiment_obj['run_id']}, url: {logged_model}")
    return experiment_obj['run_id'], experiment_obj

# helper to predict on specific dataframe
def model_predict(df_input, name_model, list_col_keep=None, model_meta=None):
    if model_meta is None:   # backwards compatible if no experiment provided - 7/23
        _, model_meta = model_lookup(name_model)
    list_prior = df_input.dtypes

    # Predict on a Spark DataFrame.
    logged_model = f"runs:/{model_meta['run_id']}/{model_meta['params.model_name']}"
    fn_log(f"[model_predict]: Retrieved valid model, run: {model_meta['run_id']}, url: {logged_model}")
    
    # Load model as a Spark UDF. Override result_type if the model does not return double values.
    model = mlflow.spark.load_model(logged_model)
    df_predict = model.transform(df_input)
    df_predict = df_predict.withColumn('model_id', F.lit(model_meta['run_id']))
    list_final = df_predict.dtypes
    
    if list_col_keep is not None:
        list_diff = [x[0] for x in list(set(list_final) - set(list_prior))]
        list_diff += list_col_keep
        df_predict = df_predict.select(list_diff)

    # return combined predictions
    return df_predict


# COMMAND ----------

# _, model_meta = model_lookup("taxi_popular")
# print(model_meta)

# COMMAND ----------

########################################################################################
#### THE CODE BELOW THIS POINT IS NOT CURRENTLY IN USE BUT IS AVAILABLE FOR INSPECTION
#### OF OTHER MODEL DEVELEOPMENT OPTIONS
########################################################################################


# COMMAND ----------

# from pyspark.ml.tuning import CrossValidatorModel
# cvModel = CrossValidatorModel.read().load('/users/ez2685/cvmodel')
# print(cvModel.extractParamMap())

# COMMAND ----------

import lightgbm as lgb

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, accuracy_score

def model_sample_balance(df, col_label, max_sample_imbalance=10):
    # apply stratified sampling
    pdf_ratio = (
        # FILTER to TRAIN / $ FILTER TO TEST        
        df.groupBy(F.col(col_label))
        .agg(F.count(col_label).alias('count'))
        .toPandas()
        .set_index(col_label)
    )
    dict_log = {'count_original': str(pdf_ratio['count'].to_dict())}
    dict_log['count_imblance_max'] = max_sample_imbalance
    fn_log(f"[model_sample_balance] Total label (original): {dict_log['count_original']}")
    num_min = pdf_ratio['count'].min()
    pdf_ratio['ratio_total'] = (num_min * max_sample_imbalance)/pdf_ratio['count']
    pdf_ratio['ratio_max'] = pdf_ratio['ratio_total'].apply(lambda x: min(1.0, x))
    dict_ratio = pdf_ratio['ratio_max'].to_dict()
    fn_log(f"[model_sample_balance] Stratified sampling: {dict_ratio}")
    df_resampled = df.sampleBy(F.col(col_label), fractions=dict_ratio, seed=42)
    return df_resampled, dict_log


def model_pandas_split(pdf_train, pdf_test, col_label, train_subsample=0.2):
    # X = X.apply(pd.to_numeric, errors='coerce')
    y = pdf_train[col_label] 
    X = pdf_train.drop([col_label], axis=1)
    if train_subsample:
        X_train, _, y_train, _ = train_test_split(X, y, shuffle=True, stratify=y, test_size=0.2, random_state=15)
    else:
        X_train = X
        y_train = y        

    y_test = pdf_test[col_label] 
    X_test = pdf_test.drop([col_label], axis=1)
    return X_train, y_train, X_test, y_test


def model_train_lightgm(X_train, y_train, X_test, y_test, early_stopping_rounds=200):
    lgb_params = { 
                        'objective':['binary'],
                        'boosting_type':'gbdt',
                        'metric':["auc", "accuracy"],
                        'n_estimators':5000,
                        'n_jobs':-1,
                        'learning_rate':0.01,
                        'num_leaves': 2**10,
                        'max_depth':20,
                        'min_data_in_leaf': 20,
                        'lambda_l1': 20,
                        'lambda_l2': 20,
                        'tree_learner':'serial',
                        'feature_fraction': 0.8, 
                        'bagging_freq':1,      
                        'bagging_fraction': 0.8, 
                        'max_bin':150,
                        'verbose':-1,
                        'random_state': 32
    } 

    # include imputing and normalizing
    transformer = model_preprocess_pipeline()    
    clf = make_pipeline(transformer, lgb.LGBMClassifier()) # use defaults? **lgb_params))
    tran1 = transformer.fit(X_train)
    X_trainpreproc = tran1.transform(X_train)
    X_testpreproc = tran1.transform(X_test)

    # update for pipeline, early stopping condition - trained via pipeline x-eval
    fit_params = {
        'lgbmclassifier__early_stopping_rounds': early_stopping_rounds,
        'lgbmclassifier__eval_metric': ["auc", "accuracy"],
        'lgbmclassifier__eval_set': [(X_trainpreproc, y_train), (X_testpreproc, y_test)]
    }
    # train quickly
    clf.fit(X_train, y_train, **fit_params)    

    train_pred = clf.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred)
    score_thresh = TunedThreshold.compute_prediction_threshold(y_train, train_pred)
    train_acc = accuracy_score(y_train, train_pred > score_thresh)
    fn_log("Training AUC is %.4f, Accuracy is %.4f"%(train_auc, train_acc))

    # test data for predictions
    test_pred = clf.predict_proba(X_test)[:, 1]
    test_bin = test_pred > score_thresh
    test_auc = roc_auc_score(y_test, test_pred)
    test_acc = accuracy_score(y_test, test_bin)
    fn_log("Testing AUC is %.4f, Accuracy is %.4f"%(test_auc, test_acc))

    return clf, score_thresh, test_pred, test_bin



# COMMAND ----------


def model_preprocess_pipeline(pdf):
    # Data preprocessing, fill NAN & remove stop words
    # TODO - use https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Imputer.html

    #tokenizing
    # conv_df['tokenized_title'] = [simple_preprocess(line, deacc=True) for line in conv_df['rmvsw_title']] 
    # conv_df['tokenized_ind'] = [simple_preprocess(line, deacc=True) for line in conv_df['rmvsw_ind']]
    
    from sklearn.preprocessing import OrdinalEncoder
    transformer_list = []
    for col_work in [x[0] for x in pdf.dtypes if type(x[1])==str]:
        fn_log(f"[model_preprocess_pipeline] Working on {col_work}...")
        pdf_uni = pdf.groupby(col_work).count().reset_index()
        print(pdf_uni)
#     enc_ord = OrdinalEncoder()

    # https://scikit-learn.org/stable/modules/impute.html#marking-imputed-values
    transformer = FeatureUnion(
        transformer_list=transformer_list + [
            ('features', SimpleImputer(strategy='mean', add_indicator=True)),
            # ('indicators', MissingIndicator())
        ]
    )
    return transformer    

