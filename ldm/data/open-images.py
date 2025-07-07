from copy import copy
import os
from random import choice as random_choice, randint as random_randint, uniform as random_uniform
import albumentations as A
from bezier import Curve
from numpy import asarray as np_asarray, asfortranarray as np_asfortranarray
import numpy as np
from PIL import ImageDraw
from PIL.Image import fromarray as Image_fromarray, new as Image_new, open as Image_open
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list.append(ToTensor())

    if normalize:
        transform_list.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return Compose(transform_list) if transform_list else transform_list[0]


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list.append(ToTensor())

    if normalize:
        transform_list.append(Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
    return Compose(transform_list) if transform_list else transform_list[0]


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
        bad_list = {
            '1af17f3d912e9aac.txt',
            '1d5ef05c8da80e31.txt',
            '3095084b358d3f2d.txt',
            '3ad7415a11ac1f5e.txt',
            '42a30d8f8fba8b40.txt',
            '1366cde3b480a15c.txt',
            '03a53ed6ab408b9f.txt'
        }

        if state == "train":
            bbox_path_list = []
            # dir_name_list=['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
            # dir_name_list = ('0',)
            dir_name_list = ('mine',)
            for dir_name in dir_name_list:
                bbox_dir=os.path.join(args['dataset_dir'],'bbox',f'train_{dir_name}')
                per_dir_file_list=os.listdir(bbox_dir)
                # for file_name in per_dir_file_list:
                #     if file_name not in bad_list:
                #         bbox_path_list.append(os.path.join(bbox_dir,file_name))
                bbox_path_list.extend([os.path.join(bbox_dir,file_name) for file_name in per_dir_file_list if file_name not in bad_list])
        elif state == "validation":
            bbox_dir = os.path.join(args['dataset_dir'],'bbox','validation')
            bbox_path_list = [os.path.join(bbox_dir,file_name) for file_name in os.listdir(bbox_dir) if file_name not in bad_list]
        else:
            bbox_dir = os.path.join(args['dataset_dir'],'bbox','test')
            bbox_path_list = [os.path.join(bbox_dir,file_name) for file_name in os.listdir(bbox_dir) if file_name not in bad_list]
        bbox_path_list.sort()
        self.length = len(bbox_path_list)
        self.bbox_path_list = bbox_path_list
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
        file_name=os.path.splitext(os.path.basename(bbox_path))[0]+'.png'
        dir_name=bbox_path.split('/')[-2]
        img_path=os.path.join('dataset/open-images/images',dir_name,file_name)

        with open(bbox_path) as f:
            bbox_list = [
                [int(float(coord)) for coord in line_stripped.split()]
                for line in f if (line_stripped := line.strip())
            ]

        x1, y1, x2, y2 = random_choice(bbox_list)
        img_p = Image_open(img_path).convert("RGB")
        try:
            img_p.verify()
        except Exception as e:
            raise ValueError(f'{img_path=} is corrupted') from e
        W, H = img_p.size

        ### Get reference image
        bbox_pad_0 = x1 - min(10, x1)
        bbox_pad_1 = y1 - min(10, y1)
        bbox_pad_2 = x2 + min(10, W-x2)
        bbox_pad_3 = y2 + min(10, H-y2)
        # img_p_np = cv2.imread(img_path)
        # img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        img_p_np = np_asarray(img_p)
        ref_image_tensor=img_p_np[bbox_pad_1:bbox_pad_3, bbox_pad_0:bbox_pad_2, :]
        ref_image_tensor=self.random_trans(image=ref_image_tensor)
        ref_image_tensor=Image_fromarray(ref_image_tensor["image"])
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
        # extended_bbox[0] = x1 - random_randint(0,int(0.4*left_freespace))
        # extended_bbox[1] = y1 - random_randint(0,int(0.4*up_freespace))
        # extended_bbox[2] = x2 + random_randint(0,int(0.4*right_freespace))
        # extended_bbox[3] = y2 + random_randint(0,int(0.4*down_freespace))
        extended_bbox = (
            x1 - random_randint(0,int(0.4*left_freespace)),
            y1 - random_randint(0,int(0.4*up_freespace)),
            x2 + random_randint(0,int(0.4*right_freespace)),
            y2 + random_randint(0,int(0.4*down_freespace)),
        )

        prob=random_uniform(0, 1)
        if prob<self.arbitrary_mask_percent:
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
            mask_img_draw=ImageDraw.Draw(mask_img)
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

        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":mask_tensor_resize,"ref_imgs":ref_image_tensor}

    def __len__(self):
        return self.length
