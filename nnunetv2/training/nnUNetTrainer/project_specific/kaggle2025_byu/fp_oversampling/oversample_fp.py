from typing import Union, Tuple, List

import numpy as np
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.nnUNetTrainer.project_specific.kaggle2025_byu.data_augmentation.more_DA import \
    MotorRegressionTrainer_BCEtopK20Loss_moreDA


from typing import Union, Tuple, List

import torch

from batchgeneratorsv2.transforms.base.basic_transform import SegOnlyTransform
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class CustomRemoveLabelTansform(SegOnlyTransform):
    def __init__(self, label_values: List[int], set_to: int, segmentation_channels: Union[int, Tuple[int, ...], List[int]] = None):
        if not isinstance(segmentation_channels, (list, tuple)) and segmentation_channels is not None:
            segmentation_channels = [segmentation_channels]
        self.segmentation_channels = segmentation_channels
        self.label_values = torch.Tensor(label_values)
        self.set_to = set_to
        super().__init__()

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        if self.segmentation_channels is None:
            channels = list(range(segmentation.shape[0]))
        else:
            channels = self.segmentation_channels
        for s in channels:
            segmentation[s][torch.isin(segmentation[s], self.label_values)] = self.set_to
        return segmentation


class MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling(MotorRegressionTrainer_BCEtopK20Loss_moreDA):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
             device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.probabilistic_oversampling = True
        self.oversample_foreground_percent = 0.4

    def get_training_transforms(
            self, patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms: ComposeTransforms = super().get_training_transforms(patch_size, rotation_for_DA, deep_supervision_scales,
                                                     mirror_axes, do_dummy_2d_data_aug, use_mask_for_norm,
                                                     is_cascaded, foreground_labels, regions, ignore_label)
        transforms.transforms.insert(-3,
                                     CustomRemoveLabelTansform(list(range(20, 30)), set_to=0))
        print(transforms)
        return transforms

    def get_validation_transforms(self,
                                  deep_supervision_scales: Union[List, Tuple, None],
                                  is_cascaded: bool = False,
                                  foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                  regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                  ignore_label: int = None,
                                  ) -> BasicTransform:
        transforms: ComposeTransforms = super().get_validation_transforms(deep_supervision_scales, is_cascaded,
                                                                          foreground_labels, regions, ignore_label)
        transforms.transforms.insert(-3,
                                     CustomRemoveLabelTansform(list(range(20, 30)), set_to=0))
        print(transforms)
        return transforms
