# EdgeVsDeep

## EMBC 2025

This repo was created to enable replication of an experiment examining the comparison between edge and deep models in EMG-based gesture recognition:  
**[Are Edge Models Sufficient for sEMG-Based Gesture Recognition?](https://github.com/deremustapha/EdgeVsDeep/tree/master/paper/EMBC2025.pdf)**  

## Table of Contents
- [Usage](#usage)
  - [Setup](#setup)
  - [Run Experiment](#experiment)


# Usage

## Setup
### 1. Create a virtual env, clone repo and install requirements

```console
$ git clone https://github.com/deremustapha/EdgeVsDeep.git
$ cd EdgeVsDeep/
$ conda activate _env_
$ pip install -r requirements.txt
```
### 2. Create a dataset folder and download DB1~DB4.
#### 2.1. **[DB1: Ninapro DB5](https://ninapro.hevs.ch/instructions/DB5.html)**  
#### 2.2. **[DB2: MyoArmBand Dataset](https://github.com/UlysseCoteAllard/MyoArmbandDataset)**  
#### 2.3. **[DB3: Hyser Dataset](https://www.physionet.org/content/hd-semg/2.0.0/pr_dataset/#files-panel)**  
#### 2.4. **[DB4: FlexEMG Dataset](https://github.com/flexemg/flexemg_v2)**  


## Run Experiment 

To execute an experiment for each dataset, first locate code. E.g For DB1 navigate to 'cd/code/DB1'. Then specify the number of subjects, define the dataset path, and indicate the save directory.
