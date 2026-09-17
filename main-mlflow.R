#to insall packages in your local renv repository
# use renv::restore()

#load the packages
source("R/loadpackages.R")
#load the functions  
source("R/functions.R")
#load the additional learners
source("R/learners.R")
#load the additional measures
source("R/measure.R")
#load the mlflow settings and functions
source("R/mlflow_miscs.R")


#Mlflow tracking settings
Sys.setenv(
  MLFLOW_TRACKING_URI = ""
)
mlflow_set_experiment(
  experiment_name = "",
)


#prepare the data
prepare_data()

#load the data
load("data/mydata.rdata")


# define the variables which will be subject to a hot-deck transformation
variables_onehot <- c(
  "HDMI",
  "TD"
)


variables_impact <- c( "BRAND")



####################
## Linear model#####
####################


#Obtain a model using as a learner a linear model
mlflow_uuid_lm <- mlflow_ML_model(
  # name of the datafranme
  data = mydata,  
  # target variable: here the log price
  target_var = "P", 
  # the time variable
  time_var = "TD",
  # the variable that identifies a product
  id = "JAN", 
  # the features that will be transformed using hotdeck encoding
  variables_onehot = variables_onehot,
  # the features that will be transformed using impact encoding
  variables_impact = variables_impact,
  # the learner that we use: here linear regression
  learner = lrn("regr.lm"),
  #the parameters of the cross-validation: folds and number of time periods 
  folds = 5,
  n= 2
) 


#Obtain imputations with the ML_model
impdata_lm <- mlflow_imputations(
  # name of the datafranme
  data = mydata,  
  # target variable: here the price
  target_var = "P", 
  # the time variable
  time_var = "TD",
  # the variable that identifies a product
  id = "JAN", 
  # the model
  model_uri = paste0("runs:/", mlflow_uuid_lm, "/rmodel")
) 


# price index calculations

index_lm <- price_index(
                        #name of the dataframe obtained with the imputations function
                        imputations = impdata_lm,
                        # target variable: here the price
                        target_var = "P", 
                        # the time variable
                        time_var = "TD",
                        #the name of the index
                        name_index = "LM"
)




####################
## Random Forest#####
####################



# Define the parameter space for random forest
search_space_ranger = ps(
  base_learner.mtry = p_int(1, 10),
  base_learner.sample.fraction = p_dbl(0.5, 1),
  base_learner.num.trees = p_int(50, 500))

#Obtain a model using Random Forest
mlflow_uuid_rf <- mlflow_ML_model(data = mydata, 
                     target_var = "P", 
                     time_var = "TD", 
                     id = "JAN",
                     variables_onehot = variables_onehot, 
                     variables_impact = variables_impact,
                     learner = lrn("regr.ranger"),
                     folds = 5,
                     n= 2,
                     search_space = search_space_ranger) 



#Obtain imputations with the ML_model
impdata_rf <- mlflow_imputations(
  # name of the datafranme
  data = mydata,  
  # target variable: here the log price
  target_var = "P", 
  # the time variable
  time_var = "TD",
  # the variable that identifies a product
  id = "JAN", 
  # the Model
  model_uri = paste0("runs:/", mlflow_uuid_rf, "/rmodel")
) 




# price index calculations
index_rf <- price_index(  imputations = impdata_rf,
                          target_var = "P", 
                          time_var = "TD",
                          name_index = "RF")





####################
##XGBoost #####
####################



# Define the parameter space for XGBoost
search_space_xgboost =ps(
  base_learner.booster           = p_fct(c("gbtree")),
  base_learner.nrounds           = p_int(16, 1000),
  base_learner.eta               = p_dbl(1e-4, 1, logscale = TRUE),
  base_learner.max_depth         = p_int(1, 20),
  base_learner.colsample_bytree  = p_dbl(1e-1, 1),
  base_learner.colsample_bylevel = p_dbl(1e-1, 1),
  base_learner.lambda            = p_dbl(1e-3, 1e3, logscale = TRUE),
  base_learner.alpha             = p_dbl(1e-3, 1e3, logscale = TRUE),
  base_learner.subsample         = p_dbl(1e-1, 1)
  
)

#run the ML pipeline with XGBoost
mlflow_uuid_xg <- mlflow_ML_model(data = mydata, 
                     target_var = "P", 
                     time_var = "TD", 
                     id = "JAN",
                     variables_onehot = variables_onehot, 
                     variables_impact = variables_impact,
                     learner = lrn("regr.xgboost"),
                     folds = 5,
                     n= 2,
                     search_space = search_space_xgboost) 



#Obtain imputations with the ML_model
impdata_xg <- mlflow_imputations(
  # name of the datafranme
  data = mydata,  
  # target variable: here the log price
  target_var = "P", 
  # the time variable
  time_var = "TD",
  # the variable that identifies a product
  id = "JAN", 
  # the Model
  model_uri = paste0("runs:/", mlflow_uuid_xg, "/rmodel")
) 



# price index calculations

index_xg <- price_index(imputations = impdata_xg,
                        target_var = "P", 
                        time_var = "TD",
                        name_index = "XG")






########################
##Closest price #####
########################



learner_cp <- LearnerClosestPrice$new(
  time_col = "TD",
  price_col = "P",
  code_col = "JAN"
)

#obtain model with closest price
mlflow_uuid_cp <- mlflow_ML_model(data = mydata, 
                     target_var = "P", 
                     time_var = "TD", 
                     id = "JAN",
                     learner = learner_cp,
                     TP_pipeline = TRUE,
                     folds = 5,
                     n= 2) 



#Obtain imputations with the ML_model
impdata_cp <- mlflow_imputations(
  # name of the datafranme
  data = mydata,  
  # target variable: here the log price
  target_var = "P", 
  # the time variable
  time_var = "TD",
  # the variable that identifies a product
  id = "JAN", 
  # the Model
  model_uri = paste0("runs:/", mlflow_uuid_cp, "/rmodel")
) 

# price index calculations
index_cp <- price_index(imputations = impdata_cp,
                        target_var = "P", 
                        time_var = "TD",
                        name_index = "CP")


########################
##    TabPFN       #####
########################

library(mlr3extralearners)
library(reticulate)

reticulate::use_python(".venv/bin/python")
py_require(c("torch", "tabpfn"))
reticulate::py_module_available("torch")
reticulate::py_module_available("tabpfn")


# set local model weight path
model_path <- "./tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt"


learner <- lrn("regr.tabpfn",
               model_path=model_path,
               device = "cuda",
               ignore_pretraining_limits = TRUE)


#run the ML pipeline with Tabpfn
mlflow_uuid_tabpfn <- mlflow_ML_model(data = mydata, 
                         target_var = "P", 
                         time_var = "TD", 
                         id = "JAN",
                         variables_onehot = variables_onehot, 
                         variables_impact = variables_impact,
                         learner = learner,
                         folds = 5,
                         n= 2) 


#Obtain imputations with the ML_model
impdata_tabpfn <- mlflow_imputations(
  # name of the datafranme
  data = mydata,  
  # target variable: here the log price
  target_var = "P", 
  # the time variable
  time_var = "TD",
  # the variable that identifies a product
  id = "JAN", 
  # the Model
  model_uri = paste0("runs:/", mlflow_uuid_tabpfn, "/rmodel")
) 



# price index calculations

index_tabpfn <- price_index(imputations = impdata_tabpfn,
                            target_var = "P", 
                            time_var = "TD",
                            name_index = "TABPFN")
