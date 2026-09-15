library("mlflow")

Sys.setenv(
  MLFLOW_PYTHON_BIN = ".venv/bin/python",
  MLFLOW_BIN = ".venv/bin/mlflow",
  MLFLOW_TRACKING_URI = ""
)

mlflow_set_experiment(
  experiment_name = "R-API-experiment",
)