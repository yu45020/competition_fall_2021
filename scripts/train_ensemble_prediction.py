import pandas as pd
from autogluon.tabular import TabularPredictor

dat_trans = pd.read_csv("data/prediction/full_train_prediction_trans.csv")
dat_gbm = pd.read_csv("data/prediction/full_train_prediction_gbm.csv")
dat_trans['y_hat_gbm'] = dat_gbm['y_hat_gbm']

gbm_opt = {  # LightGBM
    'num_boost_round': 10000,
    'boosting_type': 'gbdt',
    'two_round': False,
    'device': 'cpu',
    'extra_trees': True,
    'ag_args_fit': {'num_gpus': 0},

}

hyperparameters = {'GBM': gbm_opt}
hyperparameter_tune_kwargs = {
    'num_trials': 20,  # increasing to 50 doesn't improve results
    'scheduler': 'local',
    'searcher': 'bayesopt',
    "resource": {'num_cpus': 24, 'num_gpus': 1}
}

result_path = "checkpoints/automl_mixed_ensemble"
# model = TabularPredictor.load(result_path, verbosity=4)

model = TabularPredictor(label='y_true', problem_type='regression',
                         eval_metric='mean_squared_error', path=result_path,
                         sample_weight='auto_weight')

model = model.fit(train_data=dat_trans, tuning_data=None, time_limit=None,
                  auto_stack=True, num_bag_folds=5, num_stack_levels=3,
                  hyperparameters=hyperparameters,
                  hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                  ag_args_fit={'num_gpus': 1})
