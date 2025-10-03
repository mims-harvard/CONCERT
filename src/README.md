## Scripts to run CONCERT on Perturb-Map data  
run_concert_map.py - do counterfactual prediction on specified spots  
run_concert_map_impute.py - do imputation and imputation + counterfactual prediction on specified unseen spots  
Specify the parameters in config.yaml/config_impute.yaml
### Example commands:
<pre> python run_concert_map.py \
  --config config.yaml \
  --stage train \
  --wandb \
  --wandb_project concert-map \
  --wandb_run train

python run_concert_map_impute.py \
  --config config_impute.yaml \
  --wandb \
  --wandb_project concert-map \
  --wandb_run impute
<pre>
