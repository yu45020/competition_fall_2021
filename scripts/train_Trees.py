import autogluon.core as ag
import pandas as pd
from autogluon.tabular import TabularPredictor
from scripts.load_data_ml import load_data_split, combine_pd, load_inference_data

data_dict = load_data_split(DATA_PATH="data/train_sample.parquet",
                            var_category_type='partial', standardize=True,
                            validate_size=0.1, test_size=0.000001)
dat_train = combine_pd(data_dict, 'train')
dat_train['penalty_weight'] = data_dict['train']['weight']
dat_validate = combine_pd(data_dict, 'validate')
dat_validate['penalty_weight'] = data_dict['validate']['weight']

dat_test = combine_pd(data_dict, 'test')
dat_test['penalty_weight'] = data_dict['test']['weight']

# +++++++++++++++++++++++++++++++++++++
#           Configs
# -------------------------------------


gbm_opt = {  # LightGBM
    # default search space
    # https://github.com/awslabs/autogluon/blob/master/tabular/src/autogluon/tabular/models/lgb/hyperparameters/searchspaces.py
    'num_boost_round': 10000,
    'boosting_type': 'gbdt',
    'two_round': False,
    'device': 'cpu',
    'extra_trees': True,
    'bagging_fraction': ag.Real(lower=0.8, upper=1.0, default=1.0),
    'bagging_freq': ag.Int(lower=0, upper=60, default=0),
    'lambda_l1': ag.Real(lower=1e-10, upper=1, default=1e-10, log=True),
    'lambda_l2': ag.Real(lower=1e-10, upper=1, default=1e-10, log=True),
    # default search space
    # 'learning_rate': ag.Real(lower=5e-3, upper=0.2, default=0.05, log=True),
    # 'feature_fraction': ag.Real(lower=0.75, upper=1.0, default=1.0),
    # 'min_data_in_leaf': ag.Int(lower=2, upper=60, default=20),
    # 'num_leaves': ag.Int(lower=16, upper=96, default=31),
    'ag_args_fit': {'num_gpus': 0},

}

cat_opt = {  # CatBoost; not used
    'task_type': "GPU",
    # extra
    'border_count': ag.Int(lower=32, upper=128, default=32),
    'max_ctr_complexity': ag.Int(lower=4, upper=10, default=4),
    'random_seed': 1
}

xgb_opt = {  # XGBoost; not used
    # default search space is enough
    # https://github.com/awslabs/autogluon/blob/master/tabular/src/autogluon/tabular/models/xgboost/hyperparameters/searchspaces.py
    'tree_method': 'gpu_hist',
    'seed': 1,
    # default
    # 'n_estimators': 10000,
    # 'booster': 'gbtree',
    # 'n_jobs': -1,
    # 'learning_rate': ag.Real(lower=5e-3, upper=0.2, default=0.1, log=True),
    # 'max_depth': ag.Int(lower=3, upper=10, default=6),
    # 'min_child_weight': ag.Int(lower=1, upper=5, default=1),
    # 'gamma': ag.Real(lower=0, upper=5, default=0.01),
    # 'subsample': ag.Real(lower=0.5, upper=1.0, default=1.0),
    # 'colsample_bytree': ag.Real(lower=0.5, upper=1.0, default=1.0),
    # 'reg_alpha': ag.Real(lower=0.0, upper=10.0, default=0.0),
    # 'reg_lambda': ag.Real(lower=0.0, upper=10.0, default=1.0),
    # 'objective': 'reg:squarederror',

}
hyperparameters = {'GBM': gbm_opt}

hyperparameter_tune_kwargs = {
    'num_trials': 40,  # 20 is not enough; 50 doesn't improve results
    'scheduler': 'local',
    'searcher': 'bayesopt',
    "resource": {'num_cpus': 20, 'num_gpus': 1}
}

dat_train_val = pd.concat([dat_train, dat_validate])

result_path = "checkpoints/automl_tree_results_partial_categorical_numeric"

model = TabularPredictor(label='y', problem_type='regression', eval_metric='mean_squared_error',
                         sample_weight='penalty_weight', weight_evaluation=True, path=result_path,
                         )

model = model.fit(train_data=dat_train_val, tuning_data=None, time_limit=None,
                  auto_stack=True, num_bag_folds=5, num_stack_levels=4, num_bag_sets=40,
                  hyperparameters=hyperparameters, hyperparameter_tune_kwargs=hyperparameter_tune_kwargs)

model.refit_full('best')
model.persist_models('best', max_memory=.6)

# +++++++++++++++++++++++++++++++++++++
#           Inference On Full Train
# -------------------------------------
data_infer_dict = load_inference_data(DATA_PATH="data/train_sample.parquet",
                                      var_cat_list=data_dict['var_cat'],
                                      var_num_list=data_dict['var_num'],
                                      encoder_cat=data_dict['categorical_encoder'],
                                      encoder_num=data_dict['standardizer'],
                                      rest_set_ages=False)
dat_infer = combine_pd(data_infer_dict, 'inference')

# result_path = "checkpoints/automl_tree_results_partial_categorical_numeric"
# model = TabularPredictor.load(result_path, verbosity=4)
best_model = model.get_model_best()
prediction = model.predict(data=dat_infer, model=best_model)
dat_prediction = prediction.to_frame('y_hat_gbm')
dat_prediction['y_true'] = dat_infer['y']
dat_prediction.to_csv("data/prediction/full_train_prediction_gbm.csv", index=False)

# +++++++++++++++++++++++++++++++++++++
#           Inference On Test
# -------------------------------------
data_infer_dict = load_inference_data(DATA_PATH="data/test_sample.parquet",
                                      var_cat_list=data_dict['var_cat'],
                                      var_num_list=data_dict['var_num'],
                                      encoder_cat=data_dict['categorical_encoder'],
                                      encoder_num=data_dict['standardizer'],
                                      rest_set_ages=False)
dat_infer = combine_pd(data_infer_dict, 'inference')

# result_path = "checkpoints/automl_tree_results_partial_categorical_numeric"
# model = TabularPredictor.load(result_path, verbosity=4)
best_model = model.get_model_best()
prediction = model.predict(data=dat_infer, model=best_model)
dat_prediction = prediction.to_frame('y_hat_gbm')
dat_prediction['case_id'] = dat_infer['case_id']
dat_prediction.to_csv("data/prediction/test_prediction_gbm.csv", index=False)
