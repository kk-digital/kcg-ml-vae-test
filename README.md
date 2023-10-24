# kcg-ml-vae-test

## Training
1. Make sure dataset is in `data/`.
1. Go to `scripts/train_FCAE.py` and update hyperparameters.  
1. Run `python scripts/train_FCAE.py`.

Executing the training script will produce the following in `experiments/<save_path>`:
1. `weights/` contain the best (based on validation) and last epoch.
1. `configs.yaml` records the hyperparameters of the experiment.
1. `training.log` records the losses, this is updated in real time during training.
1. `loss.jpg` will be generated at the end of training, a plot of train and validation losses.