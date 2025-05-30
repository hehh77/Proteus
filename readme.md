#  Proteus

## A context-aware interpretable deep learning framework for multi-level single-cell spatial proteomics analysis

![Alt text](./fig1.png "Optional title")

## Hardware requirements
`Proteus` package requires only a standard computer with enough RAM and a NVIDIA GPU to support operations.
We ran the demo using the following specs:

+ CPU: 32 cores, 2.60 GHz/core
+ RAM: 64 GB
+ GPU: NVIDIA GeForce RTX 3090

## System requirements
This tool is supported for Linux. The tool has been tested on the following system:

+ Debian Linux 6.1.94-1 (kernel version 6.1.0-22) with x86_64 architecture

## Installation
To install the required packages for running Proteus, please use the following command:
```bash
conda create -n <env_name> python==3.9
conda activate <env_name>
pip install -r requirements.txt
```

### Time cost
Typical install time on a "normal" desktop computer is about 30 minutes.

## Usage
The utilization of Proteus involves a few key stages. Firstly, it is necessary to prepare and process spatial proteomics data. Following data preparation, the model training can be conducted. Once a model is trained, it can then be applied for further analysis and to explore potential insights from spatial proteomics data. Note that all the commands are run in the `code` folder.

```bash
cd ./code
```

## Dataset Pre-processing

### Data Storage and Demo Dataset
Single-cell spatial proteomics data should be placed within the ./data/ directory. For demonstration purposes, we have included the CODEX dataset, which is located in the ./data/CODEX/ folder.

### Running the Dataset Splitting Script
Once the data is in place, the dataset splitting script needs to be executed. This script prepares the data by dividing it into training and testing sets for cross-validation.

Execute the following command from the code directory:
```bash
python dataset_processing.py
```
### Expected Output
Upon script completion, the following files and folders will be generated within the ./data/CODEX/fold/ directory (when processing the CODEX dataset):

- fold_{fold}_training: Files containing the training data for each fold.
- fold_{fold}_test: Files containing the test data for each fold.
- cell_type_mapping: A file mapping cell types.

### Marker Embeddings
For the demo, pre-processed marker embeddings using ESM3, named "CODEX_marker_embeddings.npy", have been provided in the marker_embedding/ folder.


## Model Training

### Running the Training Script
Execute the following command from the code directory for model training:
```bash
python main.py
```

### Time cost
Expected run time for demo on a "normal" desktop computer is about 300 minutes. This duration may vary depending on the hardware specifications and the size of the dataset.

### Expected Output
Upon successful execution, the script will create a new directory under `./Result`. The directory will be named `CODEX_{timestamp}`, where `{timestamp}` represents the date and time of the run.
This directory will contain the following subfolders and files:

- `logs`: This folder contains the logging information for the training process. It contains training.log: Records the training progress, including metrics and other relevant information for each epoch and fold.
- `models`: This folder stores the trained model weights. It contains `fold_{fold_number}_best_model.pt`: The saved weights of the best performing model for each cross-validation fold.

## (Optional) Training on Custom Datasets: Preparation and Embedding Extraction
For custom datasets, the following guidelines apply:

### Data Formatting:
- Data must be pre-processed into a CSV file.
- Each row in the CSV is to represent a single cell.

The columns must include:

- Marker Columns: n columns, where each column represents a specific protein marker, and the corresponding cell value indicates the expression level for that marker.

- Other Information Columns: m columns containing additional information.
Required Spatial Coordinates: Columns named "X_cent" and "Y_cent" must be included to represent the spatial x and y coordinates of each cell.

- Labels for Training: It is recommended to include columns relevant to the specific analysis task, such as "cellType" or other categorical or continuous labels for model training.

### Marker Embedding Extraction with ESM3:
After data formatting, embeddings for the protein markers need to be extracted using ESM3. Please refer to the official [ESM3 tutorial and repository](https://github.com/evolutionaryscale/esm) for detailed instructions on embedding extraction
Extracted embeddings are to be saved as a .npy file.

The .npy file must contain a Python dictionary. In this dictionary:
- The keys must be strings that exactly match the marker column names in the CSV dataset.
- The values should be the corresponding numerical embedding arrays (e.g., NumPy arrays) extracted by ESM3 for each protein marker.

## Data Availability
Due to the space limitation, we used the proposed CODEX dataset as a demo dataset. The raw CODEX, MIBI1, and MIBI2 datasets, can be downloaded from the [MAPS Zenodo repository](https://zenodo.org/records/10067010). The hIntestine dataset can be downloaded from the [DRYAD repository](https://datadryad.org/landing/show?id=doi%3A10.5061%2Fdryad.g4f4qrfrc). 

