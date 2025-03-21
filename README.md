
# Adaptive Clustering Evaluation
Adaptive Clustering Evaluation (ACE) is an internal evaluation method designed to assess clustering results produced by deep clustering algorithms. It introduces a weighted scoring mechanism that combines internal scores, computed across different embedding spaces derived from the deep clustering process. This repository provides the code necessary to reproduce the results presented in the corresponding paper.["Deep Clustering Evaluation: How to Validate Internal Validation Measures"](https://arxiv.org/abs/2403.14830).



Suppose the clustering outputs consist of an embedding space (i.e., embedding data in our context) and partition outcomes, denoted as $`\phi_m = (\mathcal{Z}_m, \rho_m)`$ for $`m \in \{1, \dots, M\}`$. We consider a specific internal measure $`\pi`$ (e.g., the silhouette score) for evaluation. Let $`\pi(\rho_{m'} | \mathcal{Z}_m)`$ represent the internal score computed for the partition outcome $`\rho_{m'}`$ based on the embedding data $`\mathcal{Z}_m`$.  

To compare different partition outcomes $`\rho_1, \dots, \rho_M`$, the ACE framework assigns a score to each $`\rho_{m'}`$, where $`m' \in \{1, \dots, M\}`$, through space screening and ensemble analysis. This results in the aggregated score:  

```math
\pi(\rho_{m'} | G_s) = \sum_{m \in G_s} w^{(s)}_m \pi(\rho_{m'} | \mathcal{Z}_m)
```

where $`G_s`$ represents the selected group of spaces, and $`w^{(s)}_m`$ denotes the weight assigned to the internal score corresponding to each selected space.  

## Usage
The code for simulation and real data analysis can be found in the `./simulation/*` and `./application/*` directories, respectively.


### Set Up the Environment

First `conda` install `cython`, and then `pip` install the required dependencies in `Python 3.11` using the provided `requirement.txt`:  

```bash
conda install cython
pip install -r requirement.txt
```

Next, install the required R packages in `R 4.1`: `clusterability`, `rhdf5`, `reticulate`, `lsa`, `fpc`, `R.utils`.


This will ensure that all necessary dependencies are installed for both Python and R.


### Simulation
To run the simulation and reproduce the results presented in the paper, navigate to the `./simulation/*` directory and follow these steps:  

#### **1. Generate Simulations for Different Settings**  
Each simulation can be generated using the following command:  
```bash
python sim.py --option <simulation_setting> --cluster_std <sample_standard_deviation> --seed <random_seed> 
```  
##### **Arguments:**  
- `--option`: Specifies the simulation setting (`dense` or `sparse`).  
- `--cluster_std`: Defines the sample standard deviation.  
- `--seed`: Sets the random seed for reproducibility.  

To automate the generation of all simulations, run the script `make_sim.py`. This will create shell scripts for submitting jobs to a SLURM cluster, ensuring that all simulated datasets are generated efficiently. Simply adapt the script to match your computing environment and directory paths as needed.




#### **2. Compute Internal Scores and Dip Test Results**  
For each simulated dataset, internal scores and dip test results must be computed across all embedding spaces.  

- **Internal Scores:**  The script `calculate_metrics.py` computes internal scores for a single simulated dataset using one of the metrics reported in the main text. Run `make_metrics.py` to generate shell scripts for submitting jobs to a SLURM cluster, automating the computation of internal scores across all datasets and metrics.

- **Dip Test Results:**  The script `clusterable.R` generates the dip test results for a single simulated dataset.  Run `make_dip.py` to generate shell scripts for submitting jobs to a SLURM cluster, automating dip test computations across all datasets.  

#### **3. Compute Final Weighted Scores and Generate Plots**  
- Firstly run `ACE.py` to compute and save the final weighted scores produced by ACE.  
- Then run `boxplot.py` to generate the main figure for simulations from the saved results, as presented in the paper.


### Evaluation for Real Data Analysis

For the deep clustering evaluation conducted in the real data analysis, we provide the calculated internal measure scores, evaluated for each clustering result across different embedding spaces $`\pi(\rho_{m'} | \mathcal{Z}_m)`$ for $`m, m' \in \{1, \dots, M\}`$. These scores allow users to directly run the ACE evaluation script using them as input.

1. Download all the calculated internal and external measure values, along with other required inputs (dip test results) from the [Google Drive](https://drive.google.com/drive/folders/1MM_5UlMzpDuaezn7mI8nfamvE8TMNK5f?usp=drive_link) and save them to a local folder.

2. Run the ACE evaluation script.

To run the default setting (the ones applied in the paper), simply execute the following command:

```bash
python ACE.py 
```
For more settings, run

```bash
python ACE.py --cl_method <grouping_method> --rank_method <ranking_method> --eps <dbscan_thresh> --filter_alpha <dip_fwer> --graph_alpha <graph_fwer>
```
#### Arguments:
- `--cl_method`: Clustering method ('hdbscan', 'dbscan').
- `--rank_method`: Link analysis algorithm ('pr', 'hits').
- `--eps`: (Default: 0.05) Epsilon parameter for DBSCAN.
- `--filter_alpha`: (Default: 0.05) Family-wise error rate (FWER) for the Dip test.
- `--graph_alpha`: (Default: 0.1) FWER for creating the graph.

This evaluation script will generate a folder containing CSV files with all the rank correlation results. Then, `python table_main.py --ext <external_measure> --exp <experiment_name>` can be run to generate the main table in the main paper, where `external_measure` can be `nmi` or `acc`, corresponding to NMI or ACC, respectively, and `experiment_name` can be `hyper` or `num`, corresponding to the hyperparameter tuning and determination of cluster numbers experiments. Finally, the script `plot.py` can be run to generate the barplots.

Users can also prepare all the scores and inputs for ACE from scratch. Below, we outline the preliminary steps to generate the intermediate outputs required as inputs to ACE.

## Preliminary Steps for Evaluation

### Downloading Datasets

Download all the original datasets used to run and evaluate deep clustering algorithms from the [JULE repository](https://github.com/jwyang/JULE.torch). The contents of these datasets follow the structure below:  

| Object | Data Type | Description |
|--------|-----------|-------------|
| **File: `$DATASET/data4torch.h5`** | | |
| `data` | ndarray | Image data used for clustering |
| `labels` | ndarray | Ground truth partition labels |



### Run Deep Clustering Methods
To run the deep clustering methods *JULE* and *DEPICT* for hyperparameter tuning and cluster number determination, first download their source codes from the original repositories ([JULE repository](https://github.com/jwyang/JULE.torch); [DEPICT repository](https://github.com/herandy/DEPICT)), ensuring access to at least one GPU. Install the required dependencies as specified in the repositories, then copy the scripts from `application/script/deep_clustering/JULE` and `application/script/deep_clustering/DEPICT` into their respective downloaded folders. Finally, run `run_hyper.py` and `run_num.py` in each folder to sequentially execute all tasks across datasets and configurations designated for hyperparameter tuning and cluster number determination. Additionally, we provide the saved clustering outputs ($`\rho_{m}, \mathcal{Z}_m`$) in [Google Drive](https://drive.google.com/drive/folders/1MM_5UlMzpDuaezn7mI8nfamvE8TMNK5f?usp=drive_link). For outputs generated from running *JULE*, the embedding data $`\mathcal{Z}_m`$ is stored in a file with the format `feature{$DATASET}{$CONFIG}.h5`, and the partition outcome $`\rho_{m}`$ is stored in a file with the format `label{$DATASET}{$CONFIG}.h5`. For outputs generated from running *DEPICT*, both the embedding data $`\mathcal{Z}_m`$ and the partition outcome $`\rho_{m}`$ are stored in a file with the format `output{$DATASET}_{$CONFIG}.npz`.

The content in these files follows the structure below:

| Object | Data Type | Description |
|--------|-----------|-------------|
| **File: `feature{$DATASET}{$CONFIG}.h5`** | | |
| `feature` | ndarray | Saved embedding data |
| **File: `label{$DATASET}{$CONFIG}.h5`** | | |
| `label` | ndarray | Saved partition outcome |
| **File: `output{$DATASET}_{$CONFIG}.npz`** | | |
| `y_features` | ndarray | Saved embedding data |
| `y_pred` | ndarray | Saved partition outcome |
   


### Get the External Measure Values

Ensure all downloaded image datasets are stored in the `application/scripts/datasets` folder. Navigate to the `application/scripts` folder and run the following command to get the NMI and ACC values:

```bash
python get_truth.py --dataset <DATASET> --task <TASK>
```
For the `--dataset` argument, provide the dataset names exactly as they appear in their respective downloaded folders. For the `--task` argument, specify one of the following: `jule`, `julenum`, `DEPICT`, or `DEPICTnum`, which correspond to the two experiments—hyperparameter tuning and number cluster determination—conducted with JULE and DEPICT, respectively.

Example for the JULE hyperparameter experiment on the COIL-20 dataset:

```bash
python get_truth.py --dataset COIL-20 --task jule
```

After completing the steps, the values of the external measures are saved in a file with the format `true_{$DATASET}.pkl`. Additionally, the outputs are available in our Google Drive under the directory `$TASK/external_metric`.

### Generate Internal Measure Values
All scripts to generate internal measure values for the evaluation are in the `application/scripts/embedded_metric` folder. Since some internal measure values can only be obtained through R packages, both R and Python scripts are used. Similarly, we provide the `make.py` script to automate the execution of our scripts by generating shell scripts for cluster submission. Customize the script to fit your computing environment and directory structure as needed.

1. **Calculate Measure Values**:
   - Step 1: `embedded_data.py`: Calculates measure values for the four internal measures reported in the main paper and prepares intermediate inputs for the R script.
   - Step 2: `embedded.r`: Calculates the values for the remaining measures.
   - Step 3: `collect_embedded_metric.py`: Collect and post-processes the calculated values. Run this step after completing Step 1 and Step 2 to collect and process abnormal values from the calculated internal measure scores frome the embedding data.

2. **Generate Shell Scripts for Slurms**:
   - `make.py`: Generates shell scripts for submission to SLURM, serving as a reference for users to create their own submission scripts. This script implements the first two steps across metrics, datasets, and tasks. To switch to the second step after completing the first, simply modify Line 28 from `step = 1` to `step = 2`.

After completing the steps, all internal scores calculated based on the embedding space $`\pi(\rho_{m'} | \mathcal{Z}_m)`$ for the Calinski-Harabasz Index, Davies-Bouldin Index, and Silhouette Score (using both Euclidean and cosine distances) are saved in a file with the format `merge_{$DATASET}_{$METRIC}_score.pkl`. Scores for the other metrics are stored in a file with the format `merge_other_{$DATASET}_{$METRIC}_score.pkl`. Additionally, the outputs are available in our Google Drive under the directory `$TASK/embedded_metric`.


### Generate Raw Scores

Scripts for generating internal measure values used for the evaluation are in the `application/scripts/raw_metric` folder. Both R and Python scripts are utilized.

1. **Calculate Measure Values**:
   - `get_raw.py`: Calculates measure values for the four internal measures reported in the main paper and prepares intermediate inputs for the R script.
   - `getraw.r`: Calculates the values for the remaining measures.
   - `collect_raw.py`: Collect and post-processes the calculated values. Run this step after completing Step 1 and Step 2 to collect and process abnormal values from the calculated internal measure scores from the raw data.

2. **Generate Shell Scripts for Slurms**:
   - `make.py`: Generates shell scripts for submission to SLURM, serving as a reference for users to create their own submission scripts. This script implements the first two steps across metrics, datasets, and tasks. To switch to the second step after completing the first, simply modify Line 26 from `step = 1` to `step = 2`.

After completing the steps, all internal scores calculated based on the raw space $`\pi(\rho_{m'} | \mathcal{X})`$ for all metrics are saved in a file with the format `merge_all_{$METRIC}_score.pkl`. Additionally, the outputs are available in our Google Drive under the directory `$TASK/raw_metric`.

### Dip Test
Scripts for conducting the Dip test on embedding data derived from JULE and DEPICT are located in the `application/scripts/dip` folder. The scripts `clusterable_DEPICT.R` and `clusterable_jule.R` are specifically designed to perform Dip tests on embedding data obtained from DEPICT and JULE, respectively.  

To execute the tests, use the following commands:  
- For DEPICT embeddings:  
  ```sh
  Rscript clusterable_DEPICT.R $dataset $embedding_file1 $embedding_file2 $embedding_file3 ...
  ```  
- For JULE embeddings:  
  ```sh
  Rscript clusterable_jule.R $dataset $embedding_file1 $embedding_file2 $embedding_file3 ...
  ```  

Additionally, we provide `make_DEPICT.py` and `make_jule.py`, which generate shell scripts for submission to SLURM. These scripts facilitate running the tests across all embedding data generated for various datasets and tasks. After completing the steps, the results of the DIP test are saved in a file with the format `dip_{$DATASET}.npz`. Additionally, the outputs are available in our Google Drive under the directory `$TASK/external_metric`. Additionally, the outputs are available in our Google Drive under the directory `$TASK/dip_test`.

The structure of the files saved from the calculation of internal measure scores, external measure scores, and Dip test results is as follows:  

| Object | Data Type | Description |
|--------|-----------|-------------|
| **File: `$TASK/external_metric/true_{$DATASET}.pkl`** | | |
| `nmv` | dict | Key: Tag names of the evaluated partitioning outcomes; Value: NMI score |
| `acv` | dict | Key: Tag names of the evaluated partitioning outcomes; Value: ACC score |
| **File: `$TASK/embedded_metric/merge_{$DATASET}_{$METRIC}_score.pkl`** | | |
| `scored` | dict | Key: Metric name; Value: A dictionary where the key is the tag name of the evaluated partitioning outcome and the value is the corresponding score for the metric |
| **File: `$TASK/embedded_metric/merge_other_{$DATASET}_{$METRIC}_score.pkl`** | | |
| `scored` | dict | Key: Metric name; Value: A dictionary where the key is the tag name of the evaluated partitioning outcome and the value is the corresponding score for the metric |
| **File: `$TASK/raw_metric/merge_all_{$METRIC}_score.pkl`** | | |
| `scored` | dict | Key: Metric name; Value: A dictionary where the key is the tag name of the evaluated partitioning outcome and the value is the corresponding score for the metric |
| **File: `$TASK/dip_test/dip_{$DATASET}.npz`** | | |
| `pvalues1` | ndarray | P-values from Dip tests |
| `models` | ndarray | Tag names for tested spaces |



### Selection of Checkpoint
Scripts for selecting checkpoints for supplementary experiments are available in the `application/scripts/deepcluster` folder. To run *DeepCluster*, first download the source code from the original repository ([DeepCluster repository](https://github.com/facebookresearch/deepcluster)), and install the required dependencies. Next, copy all scripts from our folder into the downloaded repository. Finally, follow the instructions in the original DeepCluster repository to run the method. After obtaining the deep clustering outputs, the `clusterable.R` script conducts the Dip test, and `clusterable.py` generates SLURM scripts to facilitate testing across all embedding spaces. To compute the internal score of partition outcomes for embeddings from a specific epoch, run:  `python eval_dcl_val.py --id <epoch_id> --metric <internal_metric>`.  Additionally, `eval_full.py` and `plot.py` generate tables and figures for the supplementary materials, respectively.