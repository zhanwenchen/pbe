from copy import copy
import os
from os.path import basename
from random import choice as random_choice, randint as random_randint, uniform as random_uniform
from pathlib import Path
from warnings import warn
from tkinter import E
import albumentations as A
from bezier import Curve
from numpy import asarray as np_asarray, asfortranarray as np_asfortranarray, minimum as np_minimum, maximum as np_maximum
import numpy as np
from PIL.ImageDraw import Draw
from PIL.Image import fromarray as Image_fromarray, new as Image_new, open as Image_open
from torch.utils.data import Dataset
# from torchvision.ops import clip_boxes_to_image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from .bads import bads


'''
Multitote:
    dataset/multitote/v1/
        bbox/
            all/
            train/ - a copy of all/
            valid/ - a copy of all/
            test/ - a copy of all/
'''

# DIRPATH_IMAGES = 'dataset/open-images/images'
DIRPATH_IMAGES = 'dataset/multitote/v1'
DIRNAME_AFTER = 'after' # 'images'
BAD_LIST = {
    '1af17f3d912e9aac.txt',
    '1d5ef05c8da80e31.txt',
    '3095084b358d3f2d.txt',
    '3ad7415a11ac1f5e.txt',
    '42a30d8f8fba8b40.txt',
    '1366cde3b480a15c.txt',
    '03a53ed6ab408b9f.txt',
    # 'SHV1-paKivaT02-2109,ConsolidateTote,hcX0e4qxpt9,2025-05-26T18:08:13Z.Ind3,4.txt',
    # # 'SHV1-paKivaT02-2114,ConsolidateTote,hcX0928mkmr,2025-05-05T00:32:24Z.Ind5,6.txt',
    # 'SHV1-paKivaT02-2114,ConsolidateTote,hcX0928mkmr,2025-05-05T00:32:24Z.Ind5,6.txt',
    # 'SHV1-paKivaT02-2119,ConsolidateTote,hcX072jbr3z,2025-06-08T00:46:30Z.Ind1,2.txt',
    # 'SHV1-paKivaT02-2109,ConsolidateTote,hcX002ach0f,2025-06-09T18:56:34Z.Ind1,2.txt',
    # 'SHV1-paKivaT02-2119,ConsolidateTote,hcX072kwdsb,2025-05-02T04:02:59Z.Ind2,3.txt',
    # 'SHV1-paKivaT02-2119,ConsolidateTote,hcX0742ds6x,2025-05-05T08:00:11Z.Ind0,1.txt',
    # 'SHV1-paKivaT02-2119,ConsolidateTote,hcX08sh1r7h,2025-06-06T14:54:45Z.Ind1,2.txt',
    # 'SHV1-paKivaT02-2119,ConsolidateTote,hcX08t2uckb,2025-05-14T16:04:25Z.Ind2,3.txt',
    # 'SHV1-paKivaT02-2119,ConsolidateTote,hcX08w4z0yb,2025-05-27T17:55:34Z.Ind4,5.txt',
    # 'SHV1-paKivaT02-2119,ConsolidateTote,hcX08zu7yjv,2025-05-05T00:41:18Z.Ind2,3.txt',
    # 'SHV1-paKivaT02-2119,ConsolidateTote,hcX0d9rb32j,2025-05-06T17:30:49Z.Ind11,12.txt',
    # 'SHV1-paKivaT02-2119,ConsolidateTote,hcX0dkx598p,2025-05-02T20:12:37Z.Ind4,5.txt',
    # 'SHV1-paKivaT02-2125,ConsolidateTote,hcX003z4kqb,2025-05-19T00:03:46Z.Ind1,2.txt',
    # 'SHV1-paKivaT02-2125,ConsolidateTote,hcX0d7y0myx,2025-05-28T16:21:28Z.Ind2,3.txt',
    # 'SHV1-paKivaT02-2125,ConsolidateTote,hcX0d8jidqp,2025-05-27T09:24:58Z.Ind1,2.txt',
    # 'SHV1-paKivaT02-2128,ConsolidateTote,hcX07634v9d,2025-05-06T13:52:44Z.Ind3,4.txt',
}



# BAD_LIST = BAD_LIST | set(bads['test']) | set(bads['train']) | set(bads['validation'])
BAD_LIST = BAD_LIST | bads
print(f"Total bad files: {len(BAD_LIST)}")
#
def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list.append(ToTensor())

    if normalize:
        transform_list.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transform_list[0] if len(transform_list) == 1 else Compose(transform_list)


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list.append(ToTensor())

    if normalize:
        transform_list.append(Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
    return transform_list[0] if len(transform_list) == 1 else Compose(transform_list)


def clip_bboxes(bboxes, height, width):
    """
    Clips bounding boxes to image boundaries.
    Args:
        bboxes (np.ndarray): An array of bounding boxes of shape (N, 4),
            where each box is in the format [x_min, y_min, x_max, y_max].
        image_shape (tuple): The shape of the image, e.g., (height, width, ...).
    Returns:
        np.ndarray: The clipped bounding boxes of shape (N, 4).
    """
    # clipped_bboxes = bboxes.copy()
    clipped_bboxes = np_asarray(bboxes)
    # height, width = image_shape[:2]
    clipped_bboxes[[0, 2]] = clipped_bboxes[[0, 2]].clip(min=0, max=width)
    clipped_bboxes[[1, 3]] = clipped_bboxes[[1, 3]].clip(min=0, max=height)

    # clipped_bboxes[[0, 1]] = np_maximum(0, clipped_bboxes[[0, 1]])
    # clipped_bboxes[[2, 3]] = np_minimum([width, height], clipped_bboxes[[2, 3]])
    return clipped_bboxes


# def xywh_to_xyxy(xywh):
#     """
#     Convert XYWH format (x,y center point and width, height) to XYXY format (x,y top left and x,y bottom right).
#     :param xywh: [X, Y, W, H]
#     :return: [X1, Y1, X2, Y2]
#     """
#     if np_asarray(xywh).ndim > 1 or len(xywh) > 4:
#         raise ValueError('xywh format: [x1, y1, width, height]')
#     x1 = xywh[0] - xywh[2] / 2
#     y1 = xywh[1] - xywh[3] / 2
#     x2 = xywh[0] + xywh[2] / 2
#     y2 = xywh[1] + xywh[3] / 2
#     return np_asarray([int(x1), int(y1), int(x2), int(y2)])

def crop_square_from_mask(tgt: torch_Tensor, src: torch_Tensor, mask: torch_Tensor, pad_min: float = 0.10, pad_max: float = 0.40, gen: Optional[torch_Generator] = None):
    'Return tgt, src, mask cropped to a random square around the masked area.'
    nz = torch.nonzero(mask[0] > 0, as_tuple=False)
    if nz.numel() == 0:
        return tgt, src, mask

    y1x1 = nz.min(0).values
    y2x2 = nz.max(0).values
    y1, x1 = int(y1x1[0]), int(y1x1[1])
    y2, x2 = int(y2x2[0]), int(y2x2[1])
    obj_h, obj_w = y2 - y1, x2 - x1
    rand = torch.rand(1, generator=gen).item()
    side = int(max(obj_h, obj_w) * (1 + pad_min + (pad_max - pad_min) * rand))

    _, H, W = tgt.shape
    cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
    top = max(min(cy - side // 2, H - side), 0)
    left = max(min(cx - side // 2, W - side), 0)

    v_slice, h_slice = slice(top, top + side), slice(left, left + side)
    return tgt[:, v_slice, h_slice], src[:, v_slice, h_slice], mask[:, v_slice, h_slice]

# -----------------------------------------------------------------------------
# dataset class ----------------------------------------------------------------

class PBEQuadrupleDataset(Dataset):
    'Dataset yielding (source, mask, reference, target) tuples for PBE.'

    def __init__(self, csv_file: str | Path, image_size: int = 512, crop_to_square: bool = True, clip_aug: Optional[v2.Compose] = None):
        self.df = pd.read_csv(csv_file)
        self.crop = crop_to_square
        self.to_img = _get_tensor()
        self.to_mask = _get_tensor(normalise=False)
        self.to_clip = _get_tensor_clip()
        self.resize_img = Resize([image_size, image_size])
        self.resize_mask = Resize([image_size, image_size])
        self.clip_aug = clip_aug or v2.Compose([
            v2.Resize((224, 224)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=20),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.3),
        ])

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _pil_rgb(path: str | Path):
        return Image_open(path).convert('RGB')

    @staticmethod
    def _pil_mask(path: str | Path):
        return Image_open(path).convert('L')

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        tgt = self.to_img(self._pil_rgb(row['tgt']))
        src = self.to_img(self._pil_rgb(row['src']))
        mask = self.to_mask(self._pil_mask(row['mask']))
        src = src * mask

        if self.crop:
            tgt, src, mask = crop_square_from_mask(tgt, src, mask)

        tgt = self.resize_img(tgt)
        mask = (self.resize_mask(mask) > 0.5).float()
        src = self.resize_img(src)

        ref_pil = self.clip_aug(self._pil_rgb(row['ref']))
        ref = self.to_clip(ref_pil)

        return {'inpaint_image': src, 'inpaint_mask': mask, 'ref_imgs': ref, 'GT': tgt}
