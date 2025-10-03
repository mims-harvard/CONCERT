## Scripts to run CONCERT on Perturb-Map data  
run_concert_map.py - train model and conduct counterfactual prediction on specified spots  
run_concert_map_impute.py - conduct imputation and imputation + counterfactual prediction on specified unseen spots  
Specify the parameters in config.yaml/config_impute.yaml
### Example commands:
1. train model
python run_concert_map.py \
  --config config.yaml \
  --stage train \
  --wandb \
  --wandb_project concert-map \
  --wandb_run train

  
2. conduct counterfactual prediction
python run_concert_map.py \
  --config config.yaml \
  --stage eval \
  --wandb \
  --wandb_project concert-map \
  --wandb_run train
  
3. conduct imputation + CP
python run_concert_map_impute.py \ 
  --config config_impute.yaml \
  --wandb \
  --wandb_project concert-map \
  --wandb_run impute
