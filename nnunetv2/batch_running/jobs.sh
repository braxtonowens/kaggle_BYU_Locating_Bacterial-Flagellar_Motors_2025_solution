# lsf22-gpu01
screen -dm bash -c ". ~/load_env_kaggle2025_byu.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 182 3d_fullres 1 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA "
screen -dm bash -c ". ~/load_env_kaggle2025_byu.sh && CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 182 3d_fullres 2 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA "
screen -dm bash -c ". ~/load_env_kaggle2025_byu.sh && CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 182 3d_fullres 3 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA "
screen -dm bash -c ". ~/load_env_kaggle2025_byu.sh && CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 142 3d_fullres 4 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA "
screen -dm bash -c ". ~/load_env_kaggle2025_byu.sh && CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 142 3d_fullres all -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA "
screen -dm bash -c ". ~/load_env_kaggle2025_byu.sh && CUDA_VISIBLE_DEVICES=5 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA "
#screen -dm bash -c ". ~/load_env_kaggle2025_byu.sh && CUDA_VISIBLE_DEVICES=6 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_BCEtopK50Loss "
#screen -dm bash -c ". ~/load_env_kaggle2025_byu.sh && CUDA_VISIBLE_DEVICES=7 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSELoss_lr1en3 "

# launched as jobs
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=12G ". ~/load_env_kaggle2025_byu.sh && nnUNet_save_hard_preds=t CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 182 3d_fullres all -tr MotorRegressionTrainer_RegDiceBCEtopK20Loss_moreDA --c"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=12G ". ~/load_env_kaggle2025_byu.sh && nnUNet_save_hard_preds=t CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 182 3d_fullres 0 -tr MotorRegressionTrainer_RegDiceBCEtopK20Loss_moreDA --c"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=12G ". ~/load_env_kaggle2025_byu.sh && nnUNet_save_hard_preds=t CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 182 3d_fullres 1 -tr MotorRegressionTrainer_RegDiceBCEtopK20Loss_moreDA --c"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=12G ". ~/load_env_kaggle2025_byu.sh && nnUNet_save_hard_preds=t CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 182 3d_fullres 2 -tr MotorRegressionTrainer_RegDiceBCEtopK20Loss_moreDA --c"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=12G ". ~/load_env_kaggle2025_byu.sh && nnUNet_save_hard_preds=t CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 182 3d_fullres 3 -tr MotorRegressionTrainer_RegDiceBCEtopK20Loss_moreDA --c"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=12G ". ~/load_env_kaggle2025_byu.sh && nnUNet_save_hard_preds=t CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 182 3d_fullres 4 -tr MotorRegressionTrainer_RegDiceBCEtopK20Loss_moreDA --c"

bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=12G ". ~/load_env_kaggle2025_byu.sh && nnUNet_save_hard_preds=t CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 182 3d_fullres all -tr Scheduled_MotorRegressionTrainer_RegDiceBCEtopK20Loss_moreDA --c"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=12G ". ~/load_env_kaggle2025_byu.sh && nnUNet_save_hard_preds=t CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 182 3d_fullres 0 -tr Scheduled_MotorRegressionTrainer_RegDiceBCEtopK20Loss_moreDA --c"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=12G ". ~/load_env_kaggle2025_byu.sh && nnUNet_save_hard_preds=t CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 182 3d_fullres 1 -tr Scheduled_MotorRegressionTrainer_RegDiceBCEtopK20Loss_moreDA --c"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=12G ". ~/load_env_kaggle2025_byu.sh && nnUNet_save_hard_preds=t CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 182 3d_fullres 2 -tr Scheduled_MotorRegressionTrainer_RegDiceBCEtopK20Loss_moreDA --c"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=12G ". ~/load_env_kaggle2025_byu.sh && nnUNet_save_hard_preds=t CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 182 3d_fullres 3 -tr Scheduled_MotorRegressionTrainer_RegDiceBCEtopK20Loss_moreDA --c"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=12G ". ~/load_env_kaggle2025_byu.sh && nnUNet_save_hard_preds=t CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 182 3d_fullres 4 -tr Scheduled_MotorRegressionTrainer_RegDiceBCEtopK20Loss_moreDA --c"
