# PHMC-LAR
Partially Hidden Markov Chain Linear AutoRegressive model applied to Machinery Health Diagnosis

## Requirements
 py install numpy, scipy, sklearn, pickle, concurrent.futures

## CAMPSS Datasets
- train_FD001_HI_state and train_FD003_HI_state:  dataset #1 and #3 into binary
- data/cmapss-data/global_HI/ Health Indicator into csv format, one file by trajectory
- data/cmapss-data/state/ Ground-truth segmentation


## Model learning
 python3 cmapss-experiment.py  data_file output_file P D
 
with
 - data_file equals data/cmapss-data/train_FD001_HI_state or data/cmapss-data/train_FD003_HI_state
 - output_file the name of the output model file
 - P equals 0 or 1. 0 means unsupervised scheme (MSAR model) and 1 mean PHMC-LAR model
 - D  the autoregressive order

## Inference task learning
 File cmapss_inference.py

