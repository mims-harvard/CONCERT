## Scripts to run CONCERT on Perturb-Map data  
run_concert_map.py - train model and conduct counterfactual prediction on specified spots  
run_concert_map_impute.py - conduct imputation and imputation + counterfactual prediction on specified unseen spots  
Specify the parameters in config.yaml/config_impute.yaml
### Example commands:
1. **train model**:
<pre>
python src/run_concert_map.py \
  --config src/config.yaml \
  --sample GSM5808055 \
  --project_index map \
  --stage train \
  --multi_kernel_mode True \
  --data_file ../datasets/GSM5808055_data.h5 \
  --model_file model_map_GSM5808055.pt \
  --wandb \
  --wandb_project concert-map \
  --wandb_run train
</pre>
  
3. **conduct counterfactual prediction**:
<pre>
python src/run_concert_map.py  \
  --config src/config.yaml \
  --sample GSM5808055 \
  --project_index map \
  --stage eval \
  --data_file ../datasets/GSM5808055_data.h5 \
  --model_file model_map_GSM5808055.pt \
  --multi_kernel_mode True \
  --pert_cells select_cells/pert_cells_GSM5808055_patchclose_tumor_Ifngr2.txt \
  --target_cell_tissue tumor \
  --target_cell_perturbation Ifngr2
   </pre>
 
5. **conduct imputation + CP**:  
<pre>
python src/run_concert_map_impute.py \
  --config src/config_impute.yaml \
  --sample GSM5808055 \
  --project_index map \
  --multi_kernel_mode True \
  --data_file ../datasets/GSM5808055_data.h5  \
  --model_file model_map_GSM5808055.pt  \
  --pert_cells ./select_cells/GSM5808055_impute_region2.txt \
  --target_cell_tissue tumor \
  --target_cell_perturbation Ifngr2
  </pre>
