# kcg-ml-vae-test

## Training
1. Create / modify training config file in `experiment_configs`.
1. Update config file path in `LSTM_AE.ipynb` and run the code for training.

## Results
| Model                            	| Dataset Preprocessing          	| MSE (sum over 77*768) 	|
|----------------------------------	|--------------------------------	|-----------------------	|
| VAE_training_2023-10-18_23-42-39 	| Standard Normalization [-1, 1] 	| 30669.62              	|
| VAE_training_2023-10-19_08-53-19 	| MinMax Scaling [0, 1]          	| 1137.69               	|


## Issue
- 20231017 Loss decrease very slow at 45000. 