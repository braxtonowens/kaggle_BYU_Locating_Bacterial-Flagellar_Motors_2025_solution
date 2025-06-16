This repository contains our solution to the [BYU - Locating Bacterial Flagellar Motors 2025](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/overview) Kaggle challenge. We achieve a private lb F2 score of 0.87656 and with it the second place in the competition out of >1100 teams.
Our writeup can be found [here](TODO)

The repository you see here is a fork of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Please head over there to read more about it.

# Installation
Some dependencies should be installed manually:
- Install pytorch according to the instructions on the [pytorch website](https://pytorch.org/get-started/locally/). We recommend at least version 2.7. Pick the correct CUDA version for your system. Higher is better.
- Install batchgeneratorsv2 via `pip install git+https://github.com/MIC-DKFZ/batchgeneratorsv2.git@07541d7eb5a4839aa4a5e494a123f3fe69ccfd4f`

Now you can just clone this repository and install it:

```
git clone https://github.com/MIC-DKFZ/kaggle_BYU_Locating_Bacterial-Flagellar_Motors_2025_solution.git`
cd XXX
pip install -e .
```

# Inference
Download the [model weights](Todo) and extract them. 

Inference script is provided at [xxxx/yyyy](Todo). Execute it as follows:

```commandline


```

IMPORTANT: The inference script expects tomograms to be provided in the same format as the challenge uses (a series of .jpg images)

Use `-h` for more information on running the script


# Training
Here is how to reproduce the model training.

## Path setup
nnU-Net requires environment variables pointing it to raw data, preprocessed data and results. Set them with

```
export nnUNet_results=/home/isensee/nnUNet_results
export nnUNet_preprocessed=/home/isensee/nnUNet_preprocessed
export nnUNet_raw=/home/isensee/nnUNet_raw
```

RECOMMENDED: Add these lines to your `.bashrc` file (or whatever you are using)

## Dataset
1. Download the [dataset](Todo). This contains the official challengfe data (Dataset142), Bartleys data (Dataset181), our 555 additional cases (Dataset188), our corrected annotations for 142 and 181 (Dataset186) as well as a merged and final dataset that we should be using here (Dataset189). Don't worry about the many datasets. They link to each other and nothing is duplicated.
2. Extract the MIC_DKFZ_data.7z file into your `$nnUNet_raw` folder so that the DatasetsXXX folders are directly located in there.
3. 