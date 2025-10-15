CONCERT applied on gut DSS data with a single kernel for longitudinal study.

1. Train model.
<pre> 
python src/CONCERT-long/run_concert_gut.py  \
   --config src/CONCERT-long/config.yaml \
   --stage train  \
   --data_file ../datasets/processed_gut_data.h5  \
   --model_file model_colon.pt  \
   --wandb \
   --wandb_project concert-colon \
   --wandb_run train
</pre> 

2. Do perturbation prediction on the specified time points. Here we predict inflammed spots after 30, 50 and 73 days of recovery. The arguments `--pert_cells` and `--target_cell_day` indicate the day of data to perturb and the target day of transcriptomic state.
  
After 30 days:
<pre> 

 python src/CONCERT-long/run_concert_gut.py  \
  --config src/CONCERT-long/config.yaml \
  --project_index D12_30_P0 \
  --pert_cells D12 \
  --target_cell_day 30. \
  --target_cell_perturbation 0.0 \
  --stage eval  \
  --data_file ../datasets/processed_gut_data.h5  \
  --model_file model_colon.pt  \
  --wandb \
  --wandb_project concert-colon \
  --wandb_run train
</pre> 

After 50 days:
<pre> 
 python src/CONCERT-long/run_concert_gut.py  \
  --config src/CONCERT-long/config.yaml \
  --project_index D12_50_P0 \
  --pert_cells D12 \
  --target_cell_day 50. \
  --target_cell_perturbation 0.0 \
  --stage eval  \
  --data_file ../datasets/processed_gut_data.h5  \
  --model_file model_colon.pt  \
  --wandb \
  --wandb_project concert-colon \
  --wandb_run train
</pre>

After 73 days:
<pre> 
 python src/CONCERT-long/run_concert_gut.py  \
  --config src/CONCERT-long/config.yaml \
  --project_index D12_73_P0 \
  --pert_cells D12 \
  --target_cell_day 73. \
  --target_cell_perturbation 0.0 \
  --stage eval  \
  --data_file ../datasets/processed_gut_data.h5  \
  --model_file model_colon.pt  \
  --wandb \
  --wandb_project concert-colon \
  --wandb_run train
</pre> 
