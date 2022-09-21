# Databricks notebook source
# 
# This code contains helper functions for mlflow used for the 
#   2022 Machine Learning Workshop (part of the Software Symposium)!
# Settings are derived project-level constants in 'utilities/settings'
# 

# COMMAND ----------

# MAGIC %run ./settings

# COMMAND ----------

import mlflow

def databricks_mlflow_create(notebook_name):
    dbutils.fs.mkdirs(CREDENTIALS['paths']['notebooks'])
    dbutils.fs.mkdirs(f"dbfs:{CREDENTIALS['paths']['experiments']}")
    name_experiment = f"{CREDENTIALS['paths']['notebooks']}/{notebook_name}"

    #  1. create the experiment once via the notebook, like: 
    try:
        experiment_id = mlflow.create_experiment(
            name_experiment, f"dbfs:{CREDENTIALS['paths']['experiments']}/{notebook_name}")
    except Exception as e:
        print(f"Failed to create experiment, did it exist?: {e}")
    #  2. then get the experiment via this command: 
    experiment = mlflow.get_experiment_by_name(name_experiment)
    #  3. finally set the experiment via this command: 
    mlflow.set_experiment(experiment.name)
    return experiment
  
def databricks_mlflow_delete(notebook_name):
    # for now, we interpret this request to delete generated files...
    name_experiment = f"{CREDENTIALS['paths']['notebooks']}/{notebook_name}"
    experiment = mlflow.get_experiment_by_name(name_experiment)
    if experiment is not None:
        mlflow.delete_experiment(experiment.experiment_id)
        dbutils.fs.rm(f"dbfs:/{CREDENTIALS['paths']['experiments']}/{notebook_name}", True)


# COMMAND ----------

def databricks_mlflow_load(notebook_name, metric_sort="metrics.auc", tag_dict=None, load_central=True, verbose=True):
    name_experiment = f"{CREDENTIALS['paths']['notebooks']}/{notebook_name}"
    if load_central:
        name_experiment = f"/Users/ez2685@DOMAIN/MLWS22-ez2685"
        if verbose:
            fn_log(f"[databricks_mlflow_load] NOTE: You are temporarily loading models ONLY from the workshop repo ('{name_experiment}'), not your own store.  Pass `load_central`=False or alter this function to change the default behavior.")

    #  1. create the experiment once via the notebook, like: 
    try:
        experiment_obj = mlflow.get_experiment_by_name(name_experiment)
    except Exception as e:
        print(f"Failed to create experiment, did it exist?: {e}")
        return None
    #  2. set the experiment via this command: 
    # mlflow.set_experiment(experiment.name)
    pdf_experiment = mlflow.search_runs([experiment_obj.experiment_id], 
                                        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,  # filter for non-deleted only
                                        order_by=[f"{metric_sort} DESC"])
    if tag_dict is not None:
        for k in tag_dict:
            pdf_experiment[k] = pdf_experiment[k].astype('str')
            pdf_experiment = pdf_experiment[pdf_experiment[k] == str(tag_dict[k])]
    # run_target = f"runs:/{obj_experiment['run_id']}/{obj_experiment['params.model_name']}"
    return pdf_experiment
