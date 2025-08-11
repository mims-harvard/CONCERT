CONCERT with a single 3D kernel for stroke data.

1. Train model
<pre> python run_concert_3D_stroke.py  
  --stage train  
  --data_file datasets/Mouse_brain_stroke_all_data.h5  
  --model_file model_stroke.pt  
</pre> 

2. Do perturbation prediction on the specified spots. Here we predict stroke on healthy (sham) brains. The arguments `--pert_batch` and  `--target_cell_perturbation` indicate the slide to be perturbed and the target perturbation state (ICA: ischemia central area)  
<pre> python run_concert_3D_stroke.py  
  --stage infer  
  --data_file datasets/Mouse_brain_stroke_all_data.h5   
  --model_file model_stroke.pt  
  --pert_cells select_cells/pert_cells_stroke_sham1_CIA.txt  
  --pert_batch Sham1  
  --target_cell_perturbation ICA 
</pre> 
