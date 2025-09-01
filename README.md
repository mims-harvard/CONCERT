# CONCERT predicts spatial perturbation responses across tissue niches
[![ProjectPage](https://img.shields.io/badge/project-CONCERT-red)](https://zitniklab.hms.harvard.edu/projects/CONCERT/) [![CodePage](https://img.shields.io/badge/Code-GitHub-orange)](https://github.com/mims-harvard/CONCERT/) [![Data](https://img.shields.io/badge/Data-Links-purple)](https://github.com/mims-harvard/CONCERT/tree/main/datasets) ![License](https://img.shields.io/badge/license-MIT-blue)  

[Xiang Lin](https://scholar.google.com/citations?user=SKdT80YAAAAJ&hl=en), [Zhenglun Kong](https://scholar.google.com/citations?hl=en&user=XYa4NVYAAAAJ), [Soumya Ghosh](https://scholar.google.com/citations?user=GEYQenQAAAAJ&hl=en), [Manolis Kellis](https://scholar.google.com/citations?user=lsYXBx8AAAAJ&hl=en), [Marinka Zitnik](https://scholar.google.com/citations?user=YtUDgPIAAAAJ&hl=en)  

Advancements in spatial perturbation transcriptomics (SPT) have revolutionized our understanding of cellular behavior in native tissue contexts by integrating spatial and perturbation data. However, existing computational frameworks often fail to capture spatial complexities, focusing primarily on individual cell responses. To overcome this limitation, we propose CONCERT, a spatial-aware model designed to address novel counterfactual prediction (CP) tasks. CONCERT learns perturbation-specific kernels to capture various propagation patterns, enabling predicting response gene expression spatially. Beyond perturbation prediction, CONCERT uniquely enables counterfactual predictions of gene expression by switching one or more cell attributes, providing insights that are beyond the reach of current technologies. Leveraging this capability, we applied CONCERT to two additional datasets, addressing challenges that existing sequencing technologies cannot resolve. Evaluations on Perturb-map lung cancer datasets demonstrated that CONCERT consistently outperformed benchmark models, underscoring its potential to unravel spatially complex cellular mechanisms and drive therapeutic innovation.

## ⚡ Challenges from existing perturbation models
Ignoring cells' tissue context when predicting perturbations
<p align="center">
  <img src="https://github.com/mims-harvard/CONCERT/blob/main/issue1.jpg" alt="issue1" width="400"/>
</p>

## 🔥 Challenges from existing spatial perturbation sequencing technologies
<p align="center">
  <img src="https://github.com/mims-harvard/CONCERT/blob/main/issue2.jpg" alt="issue2" width="400"/>
</p>

<p align="center">
  <img src="https://github.com/mims-harvard/CONCERT/blob/main/concert2.jpg" alt="issue2" width="400"/>
</p>

## 🛠️ Functions of CONCERT
1. Predict perturbations on cells consdiering their niches (2D or 3D).
2. Predict perturbations across slides.
3. Impute missing cells with perturbation prediction.
4. Learn scopes of perturbation effects on tissue space.
5. Disentangle perturbation effects - intra or inter cells.

<p align="center">
  <img src="https://github.com/mims-harvard/CONCERT/blob/main/concert.jpg" alt="model" width="400"/>
</p>

## ⚙️ Dependencies
Python - 3.10.12  
torch==2.1.0  
scanpy==1.10.1  
sklearn==1.4.0  
scipy==1.12.0  
pandas==2.2.0  
numpy==1.23.5  

## 💻 Run CONCERT
### Example 1: train CONCERT on a single perturb-map data
1. Model training  
<pre> python run_concert_map.py  
  --stage train  
  --dataset GSM5808054_data.h5  
  --model_file model.pt  
</pre> 

2. Define the spots for counterfactual prediction. Spots can be easily selected from the Shiny APP in the select_cells.R script in folder [select_cells](./select_cells/).
  
3. Do perturbation prediction on the specified spots. Here we predict response gene expression of the spots in spots.txt with perturbagen Jak2-KO (knockout gene Jak2). Arguments `--target_cell_tissue` and `--target_cell_perturbation` are the targert cell/spot type and perturbation state for counterfactual prediction. We can do this only if the corresponding labels are provided during training stage for disentanglement.   
<pre> python run_concert_map.py  
  --stage infer  
  --model_file model.pt  
  --spots spots.txt  
  --dataset GSM5808054_data.h5  
  --target_cell_tissue tumor  
  --target_cell_perturbation jak2-KO  
</pre> 
4. Visualized the intermediate and final outputs - see folder [outputs](./outputs/)
  
#### Also see [jupyter notebook](./run_concert_perturbMap.ipynb) showing the detailed steps of running CONCERT.  

#### 🗂️ Datasets
| Dataset        | Raw data                     | Processed data |
|----------------|---------------------------------|------|
| [Perturb-Map](https://www.cell.com/cell/fulltext/S0092-8674(22)00195-7)          | [Raw](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193460)           | [Processed](https://figshare.com/articles/dataset/Datasets_-_Perturb-Map/29198468)   |
| [Colon inflammation](https://www.nature.com/articles/s41586-024-08216-z)     | [Raw](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE245316)    | [Processed](https://figshare.com/articles/dataset/Mouse_gut_inflammation_dataset/29882873)  |
| [Brain stroke](https://www.science.org/doi/abs/10.1126/scitranslmed.adg1323)  | Available from the authors of the original paper upon request     | [Processed](https://figshare.com/articles/dataset/Mouse_brain_stroke_dataset/29882900)   |

#### ⚖️ License
The code in this package is licensed under the MIT License.

#### 📝 Reference  

#### 💬 Questions
Please leave a Github issue or contact [Xiang Lin](mailto:xianglin226@gmail.com) or [Marinka Zitnik](mailto:marinka@zitnik.si)  


