#!/bin/bash
#SBATCH --job-name=spsens_aggplot
#SBATCH --partition=kempner
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH -o /n/netscratch/mzitnik_lab/Lab/xiang/CONCERT_sens/logs/aggplot_%j.out
#SBATCH -e /n/netscratch/mzitnik_lab/Lab/xiang/CONCERT_sens/logs/aggplot_%j.err
# conda's activate.d hooks reference unbound vars -> keep `set -u` off until after activate
set +u
source /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/miniconda3_new/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_env
set -eo pipefail
cd /n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval
python -u agg_sens_spac.py
python -u plot_sens_spac.py
B=/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/CONCERT-main
cp -v $B/Outputs/sensitivity/fig_sens_spac_rho.pdf  $B/manuscript/MAIN/FIG/supp/sensi_spac_rho.pdf
cp -v $B/Outputs/sensitivity/fig_sens_spac_rmse.pdf $B/manuscript/MAIN/FIG/supp/sensi_spac_rmse.pdf
echo DONE_AGGPLOT
