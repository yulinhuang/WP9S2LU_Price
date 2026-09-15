library("mlflow")

Sys.setenv(
  MLFLOW_PYTHON_BIN = ".venv/bin/python",
  MLFLOW_BIN = ".venv/bin/mlflow",
  MLFLOW_TRACKING_URI = ""
)

mlflow_set_experiment(
  experiment_name = "R-API-experiment",
)


# The function that extecutes the ML pipeline
mlflow_ML_model <- function( data,
                             target_var,
                             time_var,
                             id,
                             variables_onehot = NA,
                             variables_impact = NA,
                             learner,
                             folds = 5,
                             n = 2,
                             search_space = NA,
                             TP_pipeline = FALSE,
                             seed_value = 1234){
  
  set.seed(seed_value)
  
  # Define the regression task
  mytask <- as_task_regr(data, target = target_var)
  
  # Prepare the learner
  base_lrn <- learner$clone()
  base_lrn$predict_type <- "response"         # Ensure it produces numeric predictions
  
  # Define the pipeline
  
  if (TP_pipeline){
    #basic pipeline if basic learner
    pipeline <-   po("select",  
                     param_vals = list(selector = selector_name(c(target_var,id,time_var)))
    )%>>%
      learner
    
  }    else  {
    
    
    # --- PipeOps ---
    po_encimpact = po("encodeimpact",
                      affect_columns = selector_name(c(variables_impact))
    )
    
    po_onehot = po("encode",
                   method = "one-hot",
                   affect_columns = selector_name(c(time_var, variables_onehot))
    )
    
    po_removeid = po("select",
                     selector = selector_invert(selector_name(id))
    )
    
    po_base = po("learner_cv",
                 base_lrn,
                 id = "base_learner",
                 resampling.method = "insample"
    )
    
    po_union = po("featureunion")
    
    po_select_bias = po("select", id = "select_bias")   
    
    po_bias = po("learner", lrn("regr.lm"), id = "bias_adjustment")
    
    # --- Build Graph ---
    pipeline <- Graph$new()
    
    pipeline$add_pipeop(po_encimpact)
    pipeline$add_pipeop(po_onehot)
    pipeline$add_pipeop(po_removeid)
    pipeline$add_pipeop(po_base)
    pipeline$add_pipeop(po_union)
    pipeline$add_pipeop(po_select_bias)
    pipeline$add_pipeop(po_bias)
    
    # --- Edges ---
    pipeline$add_edge("encodeimpact", "encode")
    pipeline$add_edge("encode", "select")
    
    # encoded features → featureunion
    pipeline$add_edge("select", "featureunion")
    
    # base learner prediction → featureunion
    pipeline$add_edge("select", "base_learner")
    pipeline$add_edge("base_learner", "featureunion")
    
    # featureunion → select_bias → final regression
    pipeline$add_edge("featureunion", "select_bias")
    pipeline$add_edge("select_bias", "bias_adjustment")
    
    pipeline$pipeops$select_bias$param_set$values$selector <- selector_union(
      selector_grep("^base_learner\\."),
      selector_grep(paste0("^",time_var,"\\."))
    )
    
  }
  
  # Add the logarithmic trnasformation to the graph 
  
  g_ppl <- ppl("targettrafo", graph = pipeline)
  g_ppl$param_set$values$targetmutate.trafo <- function(x) log(x)
  g_ppl$param_set$values$targetmutate.inverter <- function(x) {
    list(response = exp(x$response))
  }
  
  # visualize and convert to a learner
  g_ppl$plot()
  mylearner <- as_learner(g_ppl)
  
  
  
  # Instantiate custom resampling 
  
  
  
  # 1) Create custom resampling and instantiate on THIS task
  resampling_method <- rsmp("custom")
  
  folds_obj <- tpcv(
    mytask,
    target_var = target_var,
    time_var   = time_var,
    id         = id,
    folds      = folds,
    n          = n
  )
  
  # Sanity checks 
  stopifnot(all(unlist(folds_obj$train_sets) %in% mytask$row_ids))
  stopifnot(all(unlist(folds_obj$test_sets)  %in% mytask$row_ids))
  
  resampling_method$instantiate(
    task       = mytask,
    train_sets = folds_obj$train_sets,
    test_sets  = folds_obj$test_sets
  )
  
  # 2) Tuning OR default modeling
  if (missing(search_space)) {
    
    # No tuning: just train the learner directly
    mylearner$train(mytask)
    tuned_learner <- mylearner
    
  } else {
    
    # Tuning with custom resampling on the same task
    ml_tuned <- AutoTuner$new(
      learner      = mylearner,
      resampling   = resampling_method,         
      measure      = msr("regr.rmse"),
      tuner        = tnr("mbo"),
      terminator   = trm("evals", n_evals = 5),
      search_space = search_space
    )
    
    # select the best hyperparameters, and refit on the full 'mytask'.
    ml_tuned$train(mytask)
    tuned_learner <- ml_tuned$learner
  }
  # Model evaluation
  
  rr <- resample(
    task = mytask,
    learner =tuned_learner,
    resampling = resampling_method,
    store_models = TRUE)
  
  
  #evaluate model performance
  
  mpe = MeasureRegrMPE$new()
  cod = MeasureRegrCOD$new()
  lmdpe = MeasureRegrLMDPE$new()
  lrmse = MeasureRegrLRMSE$new()
  mmper = MeasureRegrmmPER$new()
  
  rmse = msr("regr.rmse")
  mae = msr("regr.mae")
  mape = msr("regr.mape")
  
  model_performance <- rr$aggregate(list(rmse, mae, mape,mpe,cod,lmdpe,lrmse,mmper))
  
  #print and save model performance
  print("Model performance")
  print(model_performance)
  model_performance <- data.frame(measure= c("rmse", "mae", "mape","mpe","cod","lmdpe","lrmse","mmper"), perf = model_performance)
  save(model_performance, file=paste0("outputs/modelperformance",learner$id,".Rdata"))
  
  tuned_learner$marshal()
  
  # MLflow logging
  with(mlflow_start_run(), {
    for (i in 1:nrow(model_performance)) {
      measure <- model_performance$measure[i]    # Key (e.g., "regr.rmse")
      perf    <- model_performance$perf[i]      # Value (e.g., 2.532101e+04)
      mlflow_log_metric(measure, perf)
    }
    predictor <- carrier::crate(
      function(newdata) model$predict(newdata),
      model = tuned_learner
    )
    mlflow_set_tag("mlflow.runName", paste0("run-",learner$id))
    mlflow_log_model(predictor, artifact_path = "rmodel")
  })
  
  #return  tuned learner trained on the full data set
  return(tuned_learner)
  
}


mlflow_imputations <- function( data,
                                target_var,
                                time_var,
                                id,
                                model_uri
)
{
  
  predictor <- mlflow_load_model(model_uri)
  
  
  #define the task 
  mytask = as_task_regr(data, 
                        target = target_var)
  
  
  # Extract data and factor levels
  X = as.data.table(mytask$data())
  time_var = time_var
  time_levels = levels(X[[time_var]])
  
  
  # Make one prediction column per time level
  pred_cols = lapply(time_levels, function(tlev) {
    
    tmp = copy(X)
    
    # Overwrite the time variable for all units
    tmp[[time_var]] = factor(tlev, levels = time_levels)
    
    # Temporary task for prediction
    task_tmp = TaskRegr$new(
      id = paste0("cf_", tlev),
      backend = tmp,
      target = mytask$target_names
    )
    
    # Predict
    preds = predictor(task_tmp)
    
    # Return vector of predictions
    preds$response
  })
  
  #  Bind as columns
  pred_matrix = as.data.table(pred_cols)
  setnames(pred_matrix, paste0("pred_time_", time_levels))
  
  # Combine with original unit identifiers 
  cols <- c(id,target_var,time_var)
  final_result = cbind(X[,..cols], pred_matrix)
  
  
  #test orthogonality property
  test_predictions <- predictor(mytask)
  esp <- log(test_predictions$truth)-log(test_predictions$response)
  test_df <- data.frame (esp = esp, time = X[[time_var]])
  
  ortho_test <- tapply(test_df$esp, test_df$time, mean)
  print("Applying model on observed data: Average error by time period")
  print(ortho_test)
  rm(test_df)
  
  
  
  
  return(final_result)
}
