CONCERT applied on gut DSS data with a single kernel for multi-slides.

1. Train model.
<pre> python run_concert_gut.py \
  --config config.yaml \
  --wandb \
  --wandb_project concert-gut \
  --wandb_run train \
  --model_file gut_model.pt
</pre> 

2. Do perturbation prediction on the specified time points. Here we predict inflammed spots after 18 and 61 days of recovery. The arguments `--pert_cells` and `--target_cell_day` indicate the day of data to perturb and the target day of transcriptomic state.
<pre> python run_concert_gut.py \
  --config config.yaml \
  --stage eval \
  --wandb \
  --wandb_project concert-gut \
  --wandb_run train \
  --model_file gut_model.pt

  python python run_concert_gut.py \
  --config config.yaml \
  --stage eval \
  --target_cell_day 30 \
  --wandb \
  --wandb_project concert-gut \
  --wandb_run train \
  --model_file gut_model.pt

  python python run_concert_gut.py \
  --config config.yaml \
  --stage eval \
  --target_cell_day 50 \
  --wandb \
  --wandb_project concert-gut \
  --wandb_run train \
  --model_file gut_model.pt

  python python run_concert_gut.py \
  --config config.yaml \
  --stage eval \
  --target_cell_day 73 \
  --wandb \
  --wandb_project concert-gut \
  --wandb_run train \
  --model_file gut_model.pt
</pre> 
