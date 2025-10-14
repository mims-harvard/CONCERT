Single kernel verion of CONCERT.
### Train. 
<pre>
 python src/single_kernel/run_concert_sk.py  \
  --config src/single_kernel/config.yaml \
  --sample GSM5808054 \
  --stage train  \
  --project_index map_sk \
  --data_file ../../datasets/GSM5808054_data.h5  \
  --model_file model_GSM5808054_sk.pt  \
  --wandb \
  --wandb_project concert-map \
  --wandb_run train
</pre>

### Inference. 
<pre>
python src/single_kernel/run_concert_sk.py  \
  --config src/single_kernel/config.yaml \
  --sample GSM5808054 \
  --project_index map_sk \
  --stage eval \
  --data_file ../../datasets/GSM5808054_data.h5 \
  --model_file model_GSM5808054_sk.pt \
  --pert_cells select_cells/pert_cells_GSM5808054_patchclose_tumor_Jak2.txt \
  --target_cell_tissue tumor \
  --target_cell_perturbation Jak2
</pre>