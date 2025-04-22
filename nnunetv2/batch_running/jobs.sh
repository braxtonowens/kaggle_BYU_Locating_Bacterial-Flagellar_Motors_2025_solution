# lsf22-gpu01
screen -dm bash -c ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSETopK5Loss "
screen -dm bash -c ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSETopK10Loss "
screen -dm bash -c ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSETopK20Loss "
screen -dm bash -c ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSETopK50Loss "
screen -dm bash -c ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSETopK_2_15_Loss "
screen -dm bash -c ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=5 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSETopK_2_30_Loss "
screen -dm bash -c ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=6 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_BCELoss "
screen -dm bash -c ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=7 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSELoss_lr1en3 "

# launched as jobs
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSELoss_lr1en1 "
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSELoss_lr1en4 "
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSELoss_Adam3en4 "
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSELoss_Adam3en3 "
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSELoss_Adam3en2 "
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSELoss_Adam3en5 "

bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer "
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSERegDice3 "
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSELoss "
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_RegDice3 "
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_torch251_miniforge.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_RegDice1 "
