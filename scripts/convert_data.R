library(rio)
library(data.table)
library(arrow)

dat_train = rio::import('data/train_sample.dta',setclass = 'data.table')
dat_train[,  summary(incwage)]
dat_test = rio::import('data/test_sample.dta',setclass = "data.table")


write_parquet(dat_train, "data/train_sample.parquet") # for python
write_parquet(dat_test, "data/test_sample.parquet") # for python
 
