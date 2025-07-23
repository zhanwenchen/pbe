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


class OpenImageDataset(Dataset):
    def __init__(self,state,arbitrary_mask_percent=0,**args
        ):
        self.state=state
        self.args=args
        self.arbitrary_mask_percent=arbitrary_mask_percent
        self.kernel = np.ones((1, 1), np.uint8)
        self.random_trans=A.Compose([
            A.Resize(height=224,width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3)
        ])

        self.dataset_dir = dataset_dir = Path(args['dataset_dir'])
        # if state == "train":
        #     # bbox_path_list = []
        #     # dir_name_list=['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
        #     # dir_name_list = ('0',)
        #     # dir_name_list = ('mine',)
        #     # for dir_name in dir_name_list:
        #     bbox_dir=dataset_dir / 'bbox' / 'train'
        #     # per_dir_file_list=
        #     # for file_name in per_dir_file_list:
        #     #     if file_name not in BAD_LIST:
        #     #         bbox_path_list.append(os.path.join(bbox_dir,file_name))
        #     bbox_path_list = [os.path.join(bbox_dir,file_name) for file_name in os.listdir(bbox_dir) if file_name not in BAD_LIST]
        # # elif state == "validation":
        # else:
        bbox_dir = dataset_dir / 'bbox' / state
        self.bbox_path_list = bbox_path_list = sorted(bbox_dir / file_name for file_name in os.listdir(bbox_dir) if file_name not in BAD_LIST)
        # else:
        #     bbox_dir = os.path.join(args['dataset_dir'],'bbox','test')
        #     bbox_path_list = [os.path.join(bbox_dir,file_name) for file_name in os.listdir(bbox_dir) if file_name not in BAD_LIST]
        # bbox_path_list.sort()
        self.length = len(bbox_path_list)
        # self.bbox_path_list = bbox_path_list
        self.to_tensor_transform = get_tensor()
        self.to_tensor_clip_transform = get_tensor_clip()
        self.to_mask_tensor_transform = get_tensor(normalize=False)
        image_size = args['image_size']
        self.image_resize = Resize([image_size, image_size])
        self.mask_resize = Resize([image_size, image_size])
        # self.image_resize = Resize([image_size, image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        # self.mask_resize = Resize([image_size, image_size], interpolation=InterpolationMode.NEAREST)

    def __getitem__(self, index):
        bbox_path=self.bbox_path_list[index]
        # file_name=os.path.splitext(os.path.basename(bbox_path))[0]+'.jpg'
        file_name=f'{bbox_path.stem}.jpg'
        # file_name=os.path.splitext(os.path.basename(bbox_path))[0]+'.png'
        # breakpoint()
        # dir_name=bbox_path.split('/')[-2]
        bbox_path_parent = bbox_path.parent
        img_path = bbox_path_parent.parent.parent / DIRNAME_AFTER / bbox_path_parent.name / file_name
        # img_path=os.path.join(self.dataset_dir, DIRNAME_AFTER, dir_name, file_name)

        bbox = x1, y1, x2, y2 = [int(num) for num in bbox_path.read_text().split()]
        # with open(bbox_path) as f:
        #     bbox_list = [
        #         [int(float(coord)) for coord in line_stripped.split()]
        #         # xywh_to_xyxy([int(float(coord)) for coord in line_stripped.split()])
        #         for line in f if (line_stripped := line.strip())
        #     ]
        if not bbox or not len(bbox) == 4:
            raise ValueError(f'{bbox_path=}, {file_name=}, {img_path=}')
        # bbox = xywh_to_xyxy()

        # bbox = x1, y1, x2, y2 = random_choice(bbox_list) # [1049, 116, 1075, 238]
        img_p = Image_open(img_path).convert("RGB")
        try:
            img_p.verify()
        except Exception as e:
            raise ValueError(f'{img_path=} is corrupted') from e
        W, H = img_p.size # (1024, 670)
        # print(f'Before clipping: {bbox=}, {H=}, {W=}, {img_path=}, {basename(bbox_path)=}') # bbox = [1049, 116, 1075, 238]
        bbox = x1, y1, x2, y2 = clip_bboxes(bbox, H, W) # (np.int64(1049), np.int64(116), np.int64(1024), np.int64(238))
        # print(f'After clipping: {bbox=}, {H=}, {W=}, {img_path=}, {basename(bbox_path)=}') # bbox = [1049, 116, 1075, 238]

        ### Get reference image
        bbox_pad_0 = x1 - min(10, x1)
        bbox_pad_1 = y1 - min(10, y1)
        bbox_pad_2 = x2 + min(10, W-x2)
        bbox_pad_3 = y2 + min(10, H-y2)
        bbox_pad_0, bbox_pad_1, bbox_pad_2, bbox_pad_3 = clip_bboxes([bbox_pad_0, bbox_pad_1, bbox_pad_2, bbox_pad_3], H, W)
        # img_p_np = cv2.imread(img_path)
        # img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        img_p_np = np_asarray(img_p)
        ref_image_tensor = img_p_np[bbox_pad_1:bbox_pad_3, bbox_pad_0:bbox_pad_2, :]
        assert ref_image_tensor.size > 0, f"Array should not be empty. {img_p_np.shape=}; img_p_np[{bbox_pad_1=}:{bbox_pad_3=}, {bbox_pad_0=}:{bbox_pad_2=}, :], {img_path=}, {bbox=},\n\n{basename(bbox_path)=}"
        # try:
        # except Exception as e:
        #     print(e)
        #     # breakpoint()
            # return basename(bbox_path)
        ref_image_tensor = self.random_trans(image=ref_image_tensor)
        ref_image_tensor = Image_fromarray(ref_image_tensor["image"])
        ref_image_tensor = self.to_tensor_clip_transform(ref_image_tensor)

        ### Generate mask
        image_tensor = self.to_tensor_transform(img_p)
        # if W == 0 or H == 0:
        #     raise ValueError(f'{img_path=} is empty')

        # extended_bbox=copy(bbox)
        left_freespace=x1
        up_freespace=y1
        right_freespace=W-x2
        down_freespace=H-y2
        assert left_freespace >= 0, f'{left_freespace=} <= 0, {bbox_path=}, {img_path=}, {bbox=}, {img_p.size=}, \n\n{basename(bbox_path)=}\n'
        assert up_freespace >= 0, f'{up_freespace=} <= 0, {bbox_path=}, {img_path=}, {bbox=}, \n\n{basename(bbox_path)=}\n'
        assert right_freespace >= 0, f'{right_freespace=}=({W=}-{x2=}) <= 0, {bbox_path=}, {img_path=}, {bbox=}, {img_p.size=}, \n\n{basename(bbox_path)=}\n'
        assert down_freespace >= 0, f'{down_freespace=}=({H=}-{y2=}) <= 0, {bbox_path=}, {img_path=}, {bbox=}, {img_p.size=}, \n\n{basename(bbox_path)=}\n'
        # try:
        # except Exception as e:
        #     print(e)
        #     # breakpoint()
        #     return basename(bbox_path)
        # extended_bbox[0] = x1 - random_randint(0,int(0.4*left_freespace))
        # extended_bbox[1] = y1 - random_randint(0,int(0.4*up_freespace))
        # extended_bbox[2] = x2 + random_randint(0,int(0.4*right_freespace))
        # extended_bbox[3] = y2 + random_randint(0,int(0.4*down_freespace))
        extended_bbox = (
            x1 - random_randint(0, int(0.4*left_freespace)),
            y1 - random_randint(0, int(0.4*up_freespace)),
            x2 + random_randint(0, int(0.4*right_freespace)),
            y2 + random_randint(0, int(0.4*down_freespace)),
        )

        prob = random_uniform(0, 1)
        if prob < self.arbitrary_mask_percent:
            mask_img = Image_new('RGB', (W, H), (255, 255, 255))
            x1_x2_mid = (x1+x2)/2
            extended_bbox_mask=copy(extended_bbox)
            top_nodes = np_asfortranarray([
                            [x1, x1_x2_mid , x2],
                            [y1, extended_bbox_mask[1], y1],
                        ])
            down_nodes = np_asfortranarray([
                    [x2, x1_x2_mid , x1],
                    [y2, extended_bbox_mask[3], y2],
                ])
            left_nodes = np_asfortranarray([
                    [x1,extended_bbox_mask[0] , x1],
                    [y2, (y1+y2)/2, y1],
                ])
            right_nodes = np_asfortranarray([
                    [x2,extended_bbox_mask[2] , x2],
                    [y1, (y1+y2)/2, y2],
                ])
            top_curve = Curve(top_nodes,degree=2)
            right_curve = Curve(right_nodes,degree=2)
            down_curve = Curve(down_nodes,degree=2)
            left_curve = Curve(left_nodes,degree=2)
            pt_list=[]
            random_width=5
            for curve in (top_curve,right_curve,down_curve,left_curve):
                x_list=[]
                y_list=[]
                for i in range(1,19):
                    evaluation = curve.evaluate(i*0.05)
                    if (evaluation_0_0 := evaluation[0][0]) not in x_list and (evaluation_1_0 := evaluation[1][0]) not in y_list:
                        pt_list.append((evaluation_0_0 + random_randint(-random_width,random_width),evaluation_1_0 + random_randint(-random_width,random_width)))
                        x_list.append(evaluation_0_0)
                        y_list.append(evaluation_1_0)
            mask_img_draw=Draw(mask_img)
            mask_img_draw.polygon(pt_list,fill=(0,0,0))
            mask_tensor=self.to_mask_tensor_transform(mask_img)[0].unsqueeze(0)
        else:
            mask_img=np.zeros((H,W))
            mask_img[extended_bbox[1]:extended_bbox[3],extended_bbox[0]:extended_bbox[2]]=1
            mask_img=Image_fromarray(mask_img)
            mask_tensor=1-self.to_mask_tensor_transform(mask_img)

        ### Crop square image
        if W > H:
            left_most=extended_bbox[2]-H
            left_most = max(left_most, 0)
            right_most=extended_bbox[0]+H
            right_most = min(right_most, W)
            right_most=right_most-H
            if right_most<= left_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor
            else:
                left_pos = random_randint(left_most,right_most)
                free_space=min(extended_bbox[1],extended_bbox[0]-left_pos,left_pos+H-extended_bbox[2],H-extended_bbox[3])
                random_free_space = random_randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
                mask_tensor_cropped=mask_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
        elif W < H:
            upper_most=extended_bbox[3]-W
            upper_most = max(upper_most, 0)
            lower_most=extended_bbox[1]+W
            lower_most = min(lower_most, H) - W
            if lower_most<=upper_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor
            else:
                upper_pos = random_randint(upper_most,lower_most)
                free_space=min(extended_bbox[1]-upper_pos,extended_bbox[0],W-extended_bbox[2],upper_pos+W-extended_bbox[3])
                random_free_space = random_randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
                mask_tensor_cropped=mask_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
        else:
            image_tensor_cropped=image_tensor
            mask_tensor_cropped=mask_tensor

        image_tensor_resize=self.image_resize(image_tensor_cropped)
        mask_tensor_resize=self.mask_resize(mask_tensor_cropped)
        inpaint_tensor_resize=image_tensor_resize*mask_tensor_resize
        # breakpoint()
        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":mask_tensor_resize,"ref_imgs":ref_image_tensor}

    def __len__(self):
        return self.length
