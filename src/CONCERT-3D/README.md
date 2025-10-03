CONCERT with a single 2D/3D kernel for stroke data.
## Randomly perturbing spots or patches on healthy brain with 2D space.  
1. Train model
<pre> python run_concert_2D_stroke.py \
  --config config_2D.yaml \
  --stage train \
  --model_file stroke_2D.pt \
  --wandb \
  --wandb_project concert-stroke2D \
  --wandb_run train
</pre> 

2. Adjust sampling strategies as random or patch, also change the number of perturbed spots
<pre> python run_concert_2D_stroke.py \
  --config config_2D.yaml \
  --stage eval \
  --model_file stroke_2D.pt \
  --project_index random \ # or patch
  --pert_batch Sham1 \
  --target_cell_perturbation ICA \
  --pert_cell_number 10 # or 20, 40, 80, ...
  --wandb \
  --wandb_project concert-stroke2D \
  --wandb_run train
</pre> 

## Predicting perturbations on healthy brain with 3D space.  
1. Train model
<pre> python run_concert_3D_stroke.py \
  --config config_3D.yaml \
  --stage train \
  --model_file stroke_3D.pt \
  --wandb \
  --wandb_project concert-stroke3D \
  --wandb_run train
</pre> 

2. Do perturbation prediction on the specified spots. Here we predict stroke on healthy (sham) brains. The arguments `--pert_batch` and  `--target_cell_perturbation` indicate the slide to be perturbed and the target perturbation state (ICA: ischemia central area)  
<pre> python python run_concert_3D_map.py \
  --config config_3D.yaml \
  --stage eval \
  --model_file stroke_3D.pt \
  --pert_batch Sham1 \
  --target_cell_perturbation ICA \
  --wandb \
  --wandb_project concert-stroke3D \
  --wandb_run train
</pre> 
