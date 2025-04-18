from functools import lru_cache

import cc3d
import edt
import numpy as np
import pandas as pd
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from nnInteractive.interaction.point import build_point


class ConvertSegToRegrTarget(BasicTransform):
    def __init__(self,
                 target_type: str = 'Gaussian',
                 gaussian_sigma: float = 5,
                 edt_radius: int = 10,
                 min_segmentation_size: int = 10
                 ):
        super().__init__()
        self.target_type = target_type
        self.gaussian_sigma = gaussian_sigma
        self.edt_radius = edt_radius
        self.min_segmentation_size = min_segmentation_size
        assert target_type in ['Gaussian', 'EDT']

    def apply(self, data_dict, **params):
        seg = data_dict['segmentation']
        regr_target = torch.zeros_like(seg, dtype=torch.float32)
        assert seg.ndim == 4, f'this is only implemented for 3d and axes c, x, y, z. Got shape {seg.shape}'
        for c in range(seg.shape[0]):
            components = torch.unique(seg[c])
            components = [i for i in components if i != 0]
            if len(components) > 0:
                stats = cc3d.statistics(seg[c].numpy().astype(np.uint8))
                for ci in components:
                    bbox = stats['bounding_boxes'][ci]  # (slice(3, 9, None), slice(4, 10, None), slice(6, 12, None))
                    crop = (seg[c][bbox] == ci).numpy()
                    dist = edt.edt(crop, black_border=True)
                    center = np.unravel_index(np.argmax(dist), crop.shape)
                    center = [i + j.start for i, j in zip(center, bbox)]
                    # now place gaussian or etd on these coordinates
                    if self.target_type == 'EDT':
                        target = build_point(tuple([self.edt_radius] * 3), use_distance_transform=True, binarize=False)
                    else:
                        target = torch.from_numpy(gaussian_kernel_3d(self.gaussian_sigma))
                        target /= target.max()
                    insert_bbox = [[i - j // 2, i - j // 2 + j] for i, j in zip(center, target.shape)]
                    regr_target[c] = paste_tensor_optionalMax(regr_target[c], target, insert_bbox, use_max=True)
        # it would be nicer to write that into regression_target but that would require to change the nnunet dataloader so nah
        data_dict['segmentation'] = regr_target
        return data_dict



@lru_cache(maxsize=2)
def gaussian_kernel_3d(sigma, truncate=3.0):
    """
    Generate a 3D Gaussian kernel.

    Args:
        sigma (float or tuple): Standard deviation of the Gaussian.
        truncate (float): Truncate the filter at this many standard deviations.

    Returns:
        kernel (np.ndarray): 3D Gaussian kernel.
    """
    if isinstance(sigma, (int, float)):
        sigma = (sigma, sigma, sigma)

    # Determine kernel size (odd for symmetry)
    size = [int(truncate * s + 0.5) * 2 + 1 for s in sigma]
    z, y, x = [np.arange(-sz // 2 + 1, sz // 2 + 1) for sz in size]
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')

    kernel = np.exp(-(xx ** 2 / (2 * sigma[0] ** 2) +
                      yy ** 2 / (2 * sigma[1] ** 2) +
                      zz ** 2 / (2 * sigma[2] ** 2)))
    kernel /= kernel.sum()
    return kernel


def paste_tensor_optionalMax(target: torch.Tensor, source: torch.Tensor, bbox, use_max=False):
    """
    Paste or combine a source tensor into a target tensor using a given bounding box,
    with optional pixelwise maximum.

    Args:
        target (torch.Tensor): 3D tensor of shape (T0, T1, T2) on CPU.
        source (torch.Tensor): 3D tensor of shape (S0, S1, S2) on CPU.
                        paste_tensor_optionalMax       Must match the size of bbox.
        bbox (list or tuple): Bounding box as [[x1, x2], [y1, y2], [z1, z2]]
                              in target coordinate space.
        use_max (bool): If True, combine using pixelwise max instead of direct paste.

    Returns:
        torch.Tensor: Modified target tensor.
    """
    target_shape = target.shape
    target_indices = []
    source_indices = []

    for i, (b0, b1) in enumerate(bbox):
        t_start = max(b0, 0)
        t_end = min(b1, target_shape[i])
        if t_start >= t_end:
            return target  # No overlap

        s_start = t_start - b0
        s_end = s_start + (t_end - t_start)

        target_indices.append((t_start, t_end))
        source_indices.append((s_start, s_end))

    # Extract slices
    tz0, tz1 = target_indices[0]
    ty0, ty1 = target_indices[1]
    tx0, tx1 = target_indices[2]
    sz0, sz1 = source_indices[0]
    sy0, sy1 = source_indices[1]
    sx0, sx1 = source_indices[2]

    target_slice = target[tz0:tz1, ty0:ty1, tx0:tx1]
    source_slice = source[sz0:sz1, sy0:sy1, sx0:sx1]

    if use_max:
        target[tz0:tz1, ty0:ty1, tx0:tx1] = torch.maximum(target_slice, source_slice)
    else:
        target[tz0:tz1, ty0:ty1, tx0:tx1] = source_slice

    return target


if __name__ == '__main__':
    image = torch.zeros((1, 32, 32, 32))
    seg = torch.zeros((1, 32, 32, 32))
    seg[:, 3:9, 4:10, 6:12] = 1
    t = ConvertSegToRegrTarget('EDT', 5, 10, 10)
    ret = t(image=image, segmentation=seg)
    from batchviewer import view_batch
    view_batch(seg, ret['segmentation'])