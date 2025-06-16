This repository contains our solution to the [BYU - Locating Bacterial Flagellar Motors 2025](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/overview) Kaggle challenge. We achieve a private lb F2 score of 0.87656 and with it the second place in the competition out of >1100 teams.
Our writeup can be found [here](TODO)

The repository you see here is a fork of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Please head over there to read more about it.

# Installation
We strongly recommend installing this in a dedicated virtual environment (for example conda).
We recommend using a Linux based operating system, for example Ubuntu. Windows should work as well but is not tested.

Some dependencies should be installed manually:
- Install pytorch according to the instructions on the [pytorch website](https://pytorch.org/get-started/locally/). We recommend at least version 2.7. Pick the correct CUDA version for your system. Higher is better.
- Install batchgeneratorsv2 via `pip install git+https://github.com/MIC-DKFZ/batchgeneratorsv2.git@07541d7eb5a4839aa4a5e494a123f3fe69ccfd4f`

Now you can just clone this repository and install it:

```
git clone https://github.com/MIC-DKFZ/kaggle_BYU_Locating_Bacterial-Flagellar_Motors_2025_solution.git`
cd kaggle_BYU_Locating_Bacterial-Flagellar_Motors_2025_solution
pip install -e .
```

# Inference
Download the [model weights](Todo) and extract them. 

Inference script is provided at [nnunetv2/inference/kaggle2025_byu/inference.py](nnunetv2/inference/kaggle2025_byu/inference.py).
Provide the path to the downloaded model weights at `--ckpt-dir`

(Example provided for test folder from official competition data)
```commandline
python inference.py --input-dir /media/isensee/raw_data/bact_motors/official/test \
    --output-file /media/isensee/raw_data/bact_motors/official/test_out.csv \
    --ckpt-dir /media/isensee/data/results_nnUNet_remake/Dataset189_Kaggle2025_BYU_FlagellarMotors_mergedExternalBartleyNonBartley_512/MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25__nnUNetResEncUNetMPlans__3d_fullres_bs16_ps128_256_256 \
    --fold "('all',)"\
    --threshold 0.15\
    --gpu-id 0\
    --num-gpus 1
```

IMPORTANT: The inference script expects tomograms to be provided in the same format as the challenge uses (a series of .jpg images)

Use `-h` for more information on running the script

To use multiple GPUs for inference, take a look a this [example](nnunetv2/inference/kaggle2025_byu/multigpu_inference.sh)

# Training
Here is how to reproduce the model training.

## Path setup
nnU-Net requires environment variables pointing it to raw data, preprocessed data and results. Set them with

```
export nnUNet_results=/home/isensee/nnUNet_results
export nnUNet_preprocessed=/home/isensee/nnUNet_preprocessed
export nnUNet_raw=/home/isensee/nnUNet_raw
```
Make sure at least `$nnUNet_preprocessed` (but ideally all of them) are on a fast storage such as a local SSD or very good network drive! 

RECOMMENDED: Add these lines to your `.bashrc` file (or whatever you are using) so that the environment variables are set automatically. If you don't do this you need to export them every time you open a new terminal.


## Dataset download
1. Download the [dataset](Todo). This contains the official challenge data (Dataset142), Bartleys data (Dataset181), our 555 additional cases (Dataset188), our corrected annotations for 142 and 181 (Dataset186) as well as a merged and final dataset that we should be using here (Dataset189). All data was already resized to have the longest edge be 512 pixels and the Motors are encoded as spheres in instance segmentation maps. Don't worry about the many datasets. They link to each other and nothing is duplicated.
2. Extract the MIC_DKFZ_data.7z file into your `$nnUNet_raw` folder so that the DatasetsXXX folders are directly located in there.

## nnUNet experiment planning and preprocessing
Run the following commands (anywhere on your system)

```nnUNetv2_extract_fingerprint -d 189 -np 64```\
This will extract a 'dataset fingerprint' that nnU-Net uses for autoconfig. Set -np to a reasonable number of processes. Default here is 64. More is better but eats more RAM and I/O

```nnUNetv2_plan_experiment -d 189 -pl nnUNetPlannerResEncM```\
This will generate an automatically configured nnU-Net pipeline to use for your dataset. Usually we would just use it but for the competition we made some manual changes (larger batch and patch size and necessary adjustments to network topology).

Copy our manually adjusted nnU-Net plans from [nnunetv2/dataset_conversion/kaggle_byu/plans/nnUNetResEncUNetMPlans.json](nnunetv2/dataset_conversion/kaggle_byu/plans/nnUNetResEncUNetMPlans.json) to `$nnUNet_preprocessed/Dataset189_Kaggle2025_BYU_FlagellarMotors_mergedExternalBartleyNonBartley_512`. This should overwrite a file with the same name.

Now you can perform preprocessing:\
```nnUNetv2_preprocess -d 189 -np 64 -c 3d_fullres_bs16_ps128_256_256 -p nnUNetResEncUNetMPlans```\
Again steer the number of processes used with `-np`. Sit back and grab a cup of coffee, this may take a while

## Training
Training our final model requires 8 GPUs with at least 40GB VRAM each. Maybe 32GB will work as well - no guarantee though!

```nnUNet_n_proc_DA=24 nnUNetv2_train 189 3d_fullres_bs16_ps128_256_256 all -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25 -num_gpus 8 -p nnUNetResEncUNetMPlans```\
Note that nnUNet_n_proc_DA=24 steers the number of data augmentation workers per GPU. Adjust to your system.\
We ran our training on a node with 128C/256T. >500GB RAM needed. We had 1TB.


On 8x Nvidia A100 (40GB) PCIe (SXM will be faster) the training should take ~170s per epoch for a total of <7 days. If your training is slower than that, check for CPU or I/O bottlenecks.

Done. Your final checkpoint will be located at `$nnUNet_results/Dataset189_Kaggle2025_BYU_FlagellarMotors_mergedExternalBartleyNonBartley_512/MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25__nnUNetResEncUNetMPlans__3d_fullres_bs16_ps128_256_256`

IMPORTANT: Trainings in nnU-Net are not seeded, so you are unlikely to get exactly the same result. The result will be comparable. Maybe slightly better, maybe slightly worse. You may want to test a couple of thresholds with the newly trained model. Between 0.1 and 0.2 should be the sweet spot.

### Training with less compute
If you want to train a smaller model we recommend:

<<<<<<< HEAD
`nnUNet_n_proc_DA=24 nnUNetv2_train 189 3d_fullres all -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA -num_gpus 1 -p nnUNetResEncUNetMPlans`\
This should run ~18h on a single A100 and yield around 0.86392 private lb score with threshold 0.25
=======
```nnUNet_n_proc_DA=24 nnUNetv2_train 189 3d_fullres all -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA -num_gpus 1 -p nnUNetResEncUNetMPlans```\
This should run ~18h on a single A100 and yield around 0.86392 private lb score with threshold 0.25
>>>>>>> nnunet/challenge/kaggle2025_BYU
