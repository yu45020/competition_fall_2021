import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from scripts import models
from scripts.load_data_ml import load_data_split, load_inference_data, combine_pd

data_dict = load_data_split(DATA_PATH="data/train_sample.parquet",
                            var_category_type='all',
                            standardize=True,
                            validate_size=0.1,
                            test_size=0.00001,
                            rest_set_ages=True
                            )

model = models.TransTab(num_continuous_vars=len(data_dict['var_num']),
                        nunique_categorical_vars=data_dict['var_cat_uniqueN'],
                        dim_embedding=128,
                        dim_feedforward=512,
                        num_attn_head=8,
                        num_transformer_layers=6,
                        weight_decay=5e-5,
                        lr=6e-4,  # 8e-4 works
                        dropout_trans=0.4)
model.count_parameters()

datamodule = model.prepare_module(x_train_category=data_dict['train']['x_cat'],
                                  x_train_continuous=data_dict['train']['x_num'],
                                  y_train=data_dict['train']['y'],
                                  weight_train=data_dict['train']['weight'],
                                  x_val_category=data_dict['validate']['x_cat'],
                                  x_val_continuous=data_dict['validate']['x_num'],
                                  y_val=data_dict['validate']['y'],
                                  x_test_test_category=data_dict['test']['x_cat'],
                                  x_test_continuous=data_dict['test']['x_num'],
                                  y_test=data_dict['test']['y'],
                                  batch_size=10240,  # 8192
                                  num_workers=6
                                  )

early_stopping = model.early_stopping(patience=10, min_delta=1e-4)
checkpoint_callback = model.check_point_callback(dirpath='checkpoints/TransTab',
                                                 filename='{epoch}-{val_loss:.2f}',
                                                 monitor='val_loss', verbose=True,
                                                 every_n_epochs=1,
                                                 save_top_k=2, save_last=True,
                                                 save_on_train_epoch_end=True)

model_name = 'TransTab-full-16f-6L-128-512-w5e-5-lr6e-4-d.4-b10240'

trainer = pl.Trainer(gpus=1, accelerator='gpu',
                     # max_epochs=0,
                     enable_progress_bar=True,
                     auto_scale_batch_size=True,
                     auto_lr_find=True,
                     callbacks=[early_stopping, checkpoint_callback],
                     logger=TensorBoardLogger(save_dir="training_logs/TransTab", name=model_name),
                     log_every_n_steps=100,
                     check_val_every_n_epoch=1,
                     precision=16,
                     )

trainer.fit(model, datamodule=datamodule)

# +++++++++++++++++++++++++++++++++++++
#           Inference Full Train
# -------------------------------------
# model.freeze()
model = model.eval()
data_infer_dict = load_inference_data("data/train_sample.parquet",
                                      var_cat_list=data_dict['var_cat'],
                                      var_num_list=data_dict['var_num'],
                                      encoder_cat=data_dict['categorical_encoder'],
                                      encoder_num=data_dict['standardizer'],
                                      rest_set_ages=True)
data_inf_loader = model.prepare_dataset(data_infer_dict['inference']['x_num'],
                                        data_infer_dict['inference']['x_cat'],
                                        y=None, return_dataloader=True,
                                        batch_size=1024
                                        )
model_checkpoint = 'checkpoints/TransTab/last.ckpt'
y_hat = trainer.predict(model, dataloaders=data_inf_loader, return_predictions=True, ckpt_path=model_checkpoint)
y_hat = torch.cat(y_hat, dim=0)
y_hat = y_hat.cpu().float().numpy()
y_true = data_infer_dict['inference']['y']

y_hat_train = pd.DataFrame(np.concatenate([y_true, y_hat], axis=-1), columns=['y_true', 'y_hat_trans'])
y_hat_train.to_csv("data/prediction/full_train_prediction_trans.csv", index=True)

# +++++++++++++++++++++++++++++++++++++
#           Inference on Test
# -------------------------------------

model = model.eval()
data_infer_dict = load_inference_data("data/test_sample.parquet",
                                      var_cat_list=data_dict['var_cat'],
                                      var_num_list=data_dict['var_num'],
                                      encoder_cat=data_dict['categorical_encoder'],
                                      encoder_num=data_dict['standardizer'],
                                      rest_set_ages=True)

data_inf_loader = model.prepare_dataset(data_infer_dict['inference']['x_num'],
                                        data_infer_dict['inference']['x_cat'],
                                        y=None, return_dataloader=True,
                                        batch_size=1024
                                        )
model_checkpoint = 'checkpoints/TransTab/last.ckpt'
y_hat = trainer.predict(model, dataloaders=data_inf_loader, return_predictions=True, ckpt_path=model_checkpoint)
y_hat = torch.cat(y_hat, dim=0)
y_hat = y_hat.cpu().float().numpy()

y_hat_test = pd.DataFrame(y_hat, columns=['y_hat_trans'])
y_hat_test['case_id'] = data_infer_dict['inference']['case_id']
y_hat_test.to_csv("data/prediction/test_prediction_trans.csv", index=False)
