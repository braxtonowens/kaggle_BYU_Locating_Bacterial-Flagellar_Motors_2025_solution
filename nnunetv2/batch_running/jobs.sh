# lsf22-gpu01
screen -dm bash -c ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer"
screen -dm bash -c ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSELoss"
screen -dm bash -c ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_RegDice1"
screen -dm bash -c ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_RegDice3"
screen -dm bash -c ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSERegDice3"

# launched as jobs
bsub -R "select[hname!='e230-dgx2-2']" -R "select[hname!='e230-dgx2-1']"  -q gpu -gpu num=1:j_exclusive=yes:gmem=1G ". ~/load_env_torch224_balintsfix.sh && nnUNetv2_train 5 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
