# Report for Econometric Competition on Predicting 2019 Individual Income

The final submission is `test_submission.zip`. 

The prediction from ensemble model described in the report is `test_submission_final_ensemble_no_magic.zip`.

Preliminary analysis and final report are in the `reports` folder. 

I use R for data analysis and Python for modeling. Raw data are converted into parquet format so that both R and Python can load them. The script is located in `scripts/convert_data.R` 

Model checkpoints are not available as they are large. 

To replicate results, convert data into the `data` folder, and then run scripts  `train_*.py`



Requirements
```
Python = 3.8.3
PyTorch = 1.10.1
CUDA = 11.3
PyTorch Lightning = 1.5.7
AutoGluon = 0.3.1

1 Nvidia GPU that supports half precision, or FP 16. 

R = 4.0.3
arrow = 2.0.0
data.table = 1.14.0
rio = 0.5.16
ggpubr = 0.4.0
```