# Databricks notebook source
# MAGIC %md
# MAGIC #### This notebook demonstrates the use of an optimized data pre-processing algorithm for bias mitigation
# MAGIC 
# MAGIC - The debiasing function used is implemented in the `OptimPreproc` class.
# MAGIC - Define parameters for optimized pre-processing specific to the dataset.
# MAGIC - Divide the dataset into training, validation, and testing partitions.
# MAGIC - Learn the optimized pre-processing transformation from the training data.
# MAGIC - Train classifier on original training data.
# MAGIC - Estimate the optimal classification threshold, that maximizes balanced accuracy without fairness constraints (from the original validation set).
# MAGIC - Determine the prediction scores for original testing data. Using the estimated optimal classification threshold, compute accuracy and fairness metrics.
# MAGIC - Transform the testing set using the learned probabilistic transformation.
# MAGIC - Determine the prediction scores for transformed testing data. Using the estimated optimal classification threshold, compute accuracy and fairness metrics.

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC # Load all necessary packages
# MAGIC import sys
# MAGIC sys.path.append("../")
# MAGIC import numpy as np
# MAGIC from tqdm import tqdm
# MAGIC 
# MAGIC from aif360.datasets import BinaryLabelDataset
# MAGIC from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
# MAGIC from aif360.metrics import BinaryLabelDatasetMetric
# MAGIC from aif360.metrics import ClassificationMetric
# MAGIC from aif360.metrics.utils import compute_boolean_conditioning_vector
# MAGIC from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
# MAGIC from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
# MAGIC             import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
# MAGIC from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
# MAGIC             import get_distortion_adult, get_distortion_german, get_distortion_compas
# MAGIC from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
# MAGIC from common_utils import compute_metrics
# MAGIC 
# MAGIC from sklearn.linear_model import LogisticRegression
# MAGIC from sklearn.preprocessing import StandardScaler
# MAGIC from sklearn.metrics import accuracy_score
# MAGIC 
# MAGIC from IPython.display import Markdown, display
# MAGIC import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load dataset and specify options

# COMMAND ----------

# import dataset
dataset_used = "adult" # "adult", "german", "compas"
protected_attribute_used = 1 # 1, 2

if dataset_used == "adult":
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = load_preproc_data_adult(['sex'])
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        dataset_orig = load_preproc_data_adult(['race'])
        
    optim_options = {
        "distortion_fun": get_distortion_adult,
        "epsilon": 0.05,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
    }
    
elif dataset_used == "german":
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = load_preproc_data_german(['sex'])
        optim_options = {
            "distortion_fun": get_distortion_german,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }
    
    else:
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        dataset_orig = load_preproc_data_german(['age'])
        optim_options = {
            "distortion_fun": get_distortion_german,
            "epsilon": 0.1,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }    

elif dataset_used == "compas":
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = load_preproc_data_compas(['sex'])
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        dataset_orig = load_preproc_data_compas(['race'])
        
    optim_options = {
        "distortion_fun": get_distortion_compas,
        "epsilon": 0.05,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
    }

#random seed
np.random.seed(1)

# Split into train, validation, and test
dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Display dataset attributes

# COMMAND ----------

# print out some labels, names, etc.
display(Markdown("#### Training Dataset shape"))
print(dataset_orig_train.features.shape)
display(Markdown("#### Favorable and unfavorable labels"))
print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
display(Markdown("#### Protected attribute names"))
print(dataset_orig_train.protected_attribute_names)
display(Markdown("#### Privileged and unprivileged protected attribute values"))
print(dataset_orig_train.privileged_protected_attributes, 
      dataset_orig_train.unprivileged_protected_attributes)
display(Markdown("#### Dataset feature names"))
print(dataset_orig_train.feature_names)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Metric for original training data

# COMMAND ----------

# Metric for the original dataset
metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
display(Markdown("#### Original training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train with and transform the original training data

# COMMAND ----------

OP = OptimPreproc(OptTools, optim_options,
                  unprivileged_groups = unprivileged_groups,
                  privileged_groups = privileged_groups)

OP = OP.fit(dataset_orig_train)

# Transform training data and align features
dataset_transf_train = OP.transform(dataset_orig_train, transform_Y=True)
dataset_transf_train = dataset_orig_train.align_datasets(dataset_transf_train)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Metric with the transformed training data

# COMMAND ----------

metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                         unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
display(Markdown("#### Transformed training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())

# COMMAND ----------

# MAGIC %md
# MAGIC Optimized preprocessing has reduced the disparity in favorable outcomes between the privileged and unprivileged
# MAGIC groups (training data).

# COMMAND ----------

### Testing 
assert np.abs(metric_transf_train.mean_difference()) < np.abs(metric_orig_train.mean_difference())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load, clean up original test data and compute metric

# COMMAND ----------

dataset_orig_test = dataset_transf_train.align_datasets(dataset_orig_test)
display(Markdown("#### Testing Dataset shape"))
print(dataset_orig_test.features.shape)

metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test, 
                                         unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
display(Markdown("#### Original test dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Transform test data and compute metric

# COMMAND ----------

dataset_transf_test = OP.transform(dataset_orig_test, transform_Y = True)
dataset_transf_test = dataset_orig_test.align_datasets(dataset_transf_test)

metric_transf_test = BinaryLabelDatasetMetric(dataset_transf_test, 
                                         unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
display(Markdown("#### Transformed test dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_test.mean_difference())

# COMMAND ----------

# MAGIC %md
# MAGIC Optimized preprocessing has reduced the disparity in favorable outcomes between the privileged and unprivileged
# MAGIC groups (test data).

# COMMAND ----------

### Testing 
assert np.abs(metric_transf_test.mean_difference()) < np.abs(metric_orig_test.mean_difference())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train classifier on original data

# COMMAND ----------

# Logistic regression classifier and predictions
scale_orig = StandardScaler()
X_train = scale_orig.fit_transform(dataset_orig_train.features)
y_train = dataset_orig_train.labels.ravel()

lmod = LogisticRegression()
lmod.fit(X_train, y_train)
y_train_pred = lmod.predict(X_train)

# positive class index
pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]

dataset_orig_train_pred = dataset_orig_train.copy()
dataset_orig_train_pred.labels = y_train_pred

# COMMAND ----------

# MAGIC %md
# MAGIC #### Obtain scores original test set

# COMMAND ----------

dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
y_valid = dataset_orig_valid_pred.labels
dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
X_test = scale_orig.transform(dataset_orig_test_pred.features)
y_test = dataset_orig_test_pred.labels
dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find the optimal classification threshold from the validation set

# COMMAND ----------

num_thresh = 100
ba_arr = np.zeros(num_thresh)
class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
for idx, class_thresh in enumerate(class_thresh_arr):
    
    fav_inds = dataset_orig_valid_pred.scores > class_thresh
    dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
    dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
    
    classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                             dataset_orig_valid_pred, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    
    ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()\
                       +classified_metric_orig_valid.true_negative_rate())

best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
best_class_thresh = class_thresh_arr[best_ind]

print("Best balanced accuracy (no fairness constraints) = %.4f" % np.max(ba_arr))
print("Optimal classification threshold (no fairness constraints) = %.4f" % best_class_thresh)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predictions and fairness metrics from original test set

# COMMAND ----------

display(Markdown("#### Predictions from original testing data"))

bal_acc_arr_orig = []
disp_imp_arr_orig = []
avg_odds_diff_arr_orig = []

display(Markdown("#### Testing set"))
display(Markdown("##### Raw predictions - No fairness constraints"))

for thresh in tqdm(class_thresh_arr):
    
    fav_inds = dataset_orig_test_pred.scores > thresh
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

    if (thresh == best_class_thresh):
        disp = True
    else:
        disp = False
        
    metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred, 
                                       unprivileged_groups, privileged_groups, disp=disp)
    
    bal_acc_arr_orig.append(metric_test_bef["Balanced accuracy"])
    avg_odds_diff_arr_orig.append(metric_test_bef["Average odds difference"])
    disp_imp_arr_orig.append(metric_test_bef["Disparate impact"])

# COMMAND ----------

fig, ax1 = plt.subplots(figsize=(10,7))
ax1.plot(class_thresh_arr, bal_acc_arr_orig)
ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)


ax2 = ax1.twinx()
ax2.plot(class_thresh_arr, np.abs(1.0-np.array(disp_imp_arr_orig)), color='r')
ax2.set_ylabel('abs(1-disparate impact)', color='r', fontsize=16, fontweight='bold')
ax2.axvline(np.array(class_thresh_arr)[best_ind], 
            color='k', linestyle=':')
ax2.yaxis.set_tick_params(labelsize=14)
ax2.grid(True)

disp_imp_at_best_bal_acc_orig = np.abs(1.0-np.array(disp_imp_arr_orig))[best_ind]

# COMMAND ----------

# MAGIC %md
# MAGIC ```abs(1-disparate impact)``` must be close to zero for classifier predictions to be fair.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train classifier on transformed data and obtain predictions with its fairness metrics

# COMMAND ----------

scale_transf = StandardScaler()
X_train = scale_transf.fit_transform(dataset_transf_train.features)
y_train = dataset_transf_train.labels.ravel()

lmod = LogisticRegression()
lmod.fit(X_train, y_train)
y_train_pred = lmod.predict(X_train)

dataset_transf_train_pred = dataset_transf_train.copy()
dataset_transf_train_pred.labels = y_train_pred

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predictions and fairness metrics from transformed test set

# COMMAND ----------

dataset_transf_test_pred = dataset_transf_test.copy(deepcopy=True)
X_test = scale_transf.transform(dataset_transf_test_pred.features)
y_test = dataset_transf_test_pred.labels
dataset_transf_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

# COMMAND ----------

display(Markdown("#### Predictions from transformed testing data"))

bal_acc_arr_transf = []
disp_imp_arr_transf = []
avg_odds_diff_arr_transf = []

display(Markdown("#### Testing set"))
display(Markdown("##### Transformed predictions - No fairness constraints"))

for thresh in tqdm(class_thresh_arr):
    
    fav_inds = dataset_transf_test_pred.scores > thresh
    dataset_transf_test_pred.labels[fav_inds] = dataset_transf_test_pred.favorable_label
    dataset_transf_test_pred.labels[~fav_inds] = dataset_transf_test_pred.unfavorable_label

    if (thresh == best_class_thresh):
        disp = True
    else:
        disp = False
        
    metric_test_bef = compute_metrics(dataset_transf_test, dataset_transf_test_pred, 
                                       unprivileged_groups, privileged_groups, disp=disp)
    
    bal_acc_arr_transf.append(metric_test_bef["Balanced accuracy"])
    avg_odds_diff_arr_transf.append(metric_test_bef["Average odds difference"])
    disp_imp_arr_transf.append(metric_test_bef["Disparate impact"])

# COMMAND ----------

fig, ax1 = plt.subplots(figsize=(10,7))
ax1.plot(class_thresh_arr, bal_acc_arr_transf)
ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)


ax2 = ax1.twinx()
ax2.plot(class_thresh_arr, np.abs(1.0-np.array(disp_imp_arr_transf)), color='r')
ax2.set_ylabel('abs(1-disparate impact)', color='r', fontsize=16, fontweight='bold')
ax2.axvline(np.array(class_thresh_arr)[best_ind], 
            color='k', linestyle=':')
ax2.yaxis.set_tick_params(labelsize=14)
ax2.grid(True)

disp_imp_at_best_bal_acc_transf = np.abs(1.0-np.array(disp_imp_arr_transf))[best_ind]

# COMMAND ----------

# MAGIC %md
# MAGIC ```abs(1-disparate impact)``` must be close to zero for classifier predictions to be fair. This measure has improved using classifier trained using the transformed data compared to the original data.

# COMMAND ----------

### testing
assert disp_imp_at_best_bal_acc_transf < disp_imp_at_best_bal_acc_orig

# COMMAND ----------

# MAGIC %md
# MAGIC # Summary of Results
# MAGIC We show the optimal classification thresholds, and the fairness and accuracy metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classification Thresholds
# MAGIC 
# MAGIC | Dataset |Classification threshold|
# MAGIC |-|-|
# MAGIC |Adult|0.2674|
# MAGIC |German|0.6732|
# MAGIC |Compas|0.5148|

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fairness Metric: Disparate impact, Accuracy Metric: Balanced accuracy
# MAGIC 
# MAGIC #### Performance
# MAGIC 
# MAGIC | Dataset |Sex (Acc-Bef)|Sex (Acc-Aft)|Sex (Fair-Bef)|Sex (Fair-Aft)|Race/Age (Acc-Bef)|Race/Age (Acc-Aft)|Race/Age (Fair-Bef)|Race/Age (Fair-Aft)|
# MAGIC |-|-|-|-|-|-|-|-|-|
# MAGIC |Adult (Test)|0.7417|0.7021|0.2774|0.7729|0.7417|0.7408|0.4423|0.7645|
# MAGIC |German (Test)|0.6524|0.5698|0.9948|1.0664|0.6524|0.6067|0.3824|0.8228|
# MAGIC |Compas (Test)|0.6774|0.6606|0.6631|0.8085|0.6774|0.6790|0.6600|0.8430|

# COMMAND ----------


