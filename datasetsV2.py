# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mostly copy-paste from https://github.com/facebookresearch/deit/blob/main/datasets.py
"""

import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torch.utils.data import Dataset, DataLoader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from PIL import Image
import random
import pandas as pd
import torch
import torch.nn as nn
# import torchvision.transforms as transforms
import torch.nn.functional as F 
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, \
    GaussianBlur, Rotate, Normalize, ImageOnlyTransform

from albumentations.pytorch import ToTensorV2
from transforms.albu import IsotropicResize
import cv2
import numpy as np
from tqdm import tqdm
from torchvision.transforms import functional as TF


class SRMConv2dTransform(nn.Module):
    def __init__(self, inc=3, learnable=False):
        super(SRMConv2dTransform, self).__init__()
        self.inc = inc
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)

    def forward(self, img):
        '''
        img: single image (H, W, 3) as PIL Image
        '''
        # Convert image to tensor if it's not already
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img)
        
        # Add batch dimension
        img = img.unsqueeze(0)
        
        # Split image into three channels
        red_channel = img[:, 0:1, :, :]
        green_channel = img[:, 1:2, :, :]
        blue_channel = img[:, 2:3, :, :]

        # Apply different SRM filters to each channel
        red_out = F.conv2d(red_channel, self.kernel[0:1, :, :, :], stride=1, padding=2)
        green_out = F.conv2d(green_channel, self.kernel[0:1, :, :, :], stride=1, padding=2)
        blue_out = F.conv2d(blue_channel, self.kernel[0:1, :, :, :], stride=1, padding=2)

        # Concatenate the channels back together
        out = torch.cat((red_out, green_out, blue_out), dim=1)
        out = self.truc(out)

        # Remove batch dimension
        out = out.squeeze(0)
        
        # Convert back to PIL Image
        out = TF.to_pil_image(out)
        
        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, -2, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        filters = [
            filter1,
            filter2,
            filter3
            ]
        filters = np.array(filters)
        filters = filters[:, np.newaxis, :, :]  # Add channel dimension to filters
        filters = torch.FloatTensor(filters)    # (3,1,5,5)
        return filters

class AlbumentationsSRMConv2d(ImageOnlyTransform):
    def __init__(self, inc=3, learnable=False, always_apply=False, p=1.0):
        super(AlbumentationsSRMConv2d, self).__init__(always_apply, p)
        self.transform = SRMConv2dTransform(inc=inc, learnable=learnable)

    def apply(self, img, **params):
        img = Image.fromarray(img)
        img = self.transform(img)
        return np.array(img)


class SobelFilter(nn.Module):
    def __init__(self):
        super(SobelFilter, self).__init__()
        self.sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    def forward(self, img):
        # img should be a 4D tensor: [batch_size, channels, height, width]
        batch_size, channels, height, width = img.shape
        sobel_x = F.conv2d(img, self.sobel_kernel_x.repeat(channels, 1, 1, 1), padding=1, groups=channels)
        sobel_y = F.conv2d(img, self.sobel_kernel_y.repeat(channels, 1, 1, 1), padding=1, groups=channels)
        sobel = torch.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return sobel


class SobelFilterTransform(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(SobelFilterTransform, self).__init__(always_apply, p)
        self.sobel_filter = SobelFilter()
    
    def apply(self, img, **params):
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()  # Convert image to tensor
        sobel_img = self.sobel_filter(img_tensor).squeeze(0).permute(1, 2, 0).numpy()  # Apply Sobel filter and convert back to numpy
        
        # Ensure the image is in uint8 format
        sobel_img = np.clip(sobel_img, 0, 255).astype(np.uint8)
        
        return sobel_img


def create_train_transforms(size):
        return Compose([
            # AlbumentationsSRMConv2d(inc=3, learnable=False),
            # SobelFilterTransform(p=1.0),
            # ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
            # GaussNoise(p=0.3),
            # GaussianBlur(blur_limit=3, p=0.05),
            HorizontalFlip(),
            OneOf([
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
            OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.4),
            # ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
        )
        
def create_val_transform(size):
        return Compose([
            # AlbumentationsSRMConv2d(inc=3, learnable=False),
            # SobelFilterTransform(p=1.0),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
            # ToGray(p=0.2),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])




class DFDCDataset(Dataset):
    def __init__(self, is_train, root_dir, transform=None, apply_sobel=False, image_size=(224, 224), filter=1, args=None):
        self.root_dir = root_dir  # Update this with the root directory containing both train and validation directories
        self.transform = create_train_transforms(image_size[0]) if is_train else create_val_transform(image_size[0])
        self.is_train = is_train
        self.apply_sobel = apply_sobel
        self.image_size = image_size
        self.video_dir = ''
        self.image_path = []
        self.train_labels = np.array([])
        self.val_labels = np.array([])
        

        csv_file = os.path.join(root_dir, 'train/labels.csv')
        
        directory = 'train/faces/'
        if not self.is_train:
            print("Carregando VAL DATASET")
            csv_file = os.path.join(root_dir, 'val/labels.csv')
            directory = 'val/faces/'  # Update this with the path to the train directory
        # print(csv_file)
        self.df = pd.read_csv(csv_file)
        train_or_val = "train" if is_train else "val"
        if args.is_experiment:
            print("Loading a small part of the dataset for fast experiments")
            size_df = len(self.df)
            perc_cut = int(size_df * 0.55 )
            # perc_cut = int(size_df * 0.999)
            self.df = self.df[perc_cut:]
            print("Balancing classes..")
            # Contar a quantidade de cada rótulo
            label_counts = self.df['label'].value_counts()
            print(f"Quantidade de rótulos antes do balanceamento:\n{label_counts}")

            # Identificar o rótulo minoritário e sua quantidade
            min_label = label_counts.idxmin()
            min_count = label_counts.min()

            # Separar o DataFrame em dois, um para cada rótulo
            df_label_0 = self.df[self.df['label'] == 0]
            df_label_1 = self.df[self.df['label'] == 1]

            # Subamostrar o rótulo majoritário para igualar a quantidade do rótulo minoritário
            if len(df_label_0) > len(df_label_1):
                df_label_0 = df_label_0.sample(min_count, random_state=42)
            else:
                df_label_1 = df_label_1.sample(min_count, random_state=42)

            # Combinar os dois DataFrames de volta
            self.df = pd.concat([df_label_0, df_label_1], axis=0).reset_index(drop=True)

            # Verificar a quantidade de rótulos após o balanceamento
            label_counts_balanced = self.df['label'].value_counts()
            print(f"Quantidade de rótulos após o balanceamento:\n{label_counts_balanced}")

            if is_train:
                self.train_labels = self.df['label'].to_numpy()
            else:
                self.val_labels = self.df['label'].to_numpy()
            
            print("Saving used samples..")
            self.df.to_csv(args.output_dir+f"/{train_or_val}_images_used.csv", sep=",")


        # print(self.df)
        self.weight = len(self.df.loc[self.df['label'] == 1])
        self.total = len(self.df.loc[self.df['label'] == 0])

        self.video_dir = os.path.join(self.root_dir, directory)
        
        self.image_path = [os.path.join(self.video_dir, x) for x in tqdm(self.df['filename'], desc=f"Loading {train_or_val} dataset")]
        # print(self.image_path[0])

    def __len__(self):
        return len(self.df) 

    def __getitem__(self, idx):
        
        video_name = self.image_path[idx]

        # Check if directory exists
        if not os.path.exists(self.video_dir):
            print(f"Warning: Directory for video '{video_name}' does not exist.")
            return
        
        selected_file = video_name
        if selected_file:
            img_path = os.path.join(str(self.video_dir), selected_file)
            if os.path.isfile(img_path):
                image = Image.open(img_path).convert("RGB")
                image = image.resize(self.image_size)  # Resize image
                image = np.array(image) 
                

                if self.transform:
                    augmented = self.transform(image=image)
                    image = augmented["image"]
                

                if not isinstance(image, torch.Tensor):
                    image = torch.tensor(image, dtype=torch.float32)
    
                label = int(self.df.iloc[idx, 1])
                self.weight += int(self.df.iloc[idx, 1])
                return image, torch.tensor(label, dtype=torch.long)

        # If no image was loaded, return empty tensor for image and dummy label
        return torch.zeros(3, self.image_size[0], self.image_size[1]), torch.tensor(0, dtype=torch.long)


def build_dataset(is_train, args):
        
    dataset = DFDCDataset(root_dir='/home/eferreira/master/custom_vit/datasets/dfdc',
                            is_train=is_train,
                            # transform=transform,
                            args=args)

    return dataset, args.nb_classes


def get_transform_to_eval(size):
    return Compose([
        AlbumentationsSRMConv2d(inc=3, learnable=False),
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_transform_to_eval_SRM(size):
    return Compose([
        AlbumentationsSRMConv2d(inc=3, learnable=False),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

class SRMConv2dTransformOnlyFilter3(nn.Module):
    def __init__(self, inc=3, learnable=False):
        super(SRMConv2dTransformOnlyFilter3, self).__init__()
        self.inc = inc
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)

    def forward(self, img):
        '''
        img: single image (H, W, 3) as PIL Image
        '''
        # Convert image to tensor if it's not already
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img)
        
        # Add batch dimension
        img = img.unsqueeze(0)
        
        # Split image into three channels
        red_channel = img[:, 0:1, :, :]
        green_channel = img[:, 1:2, :, :]
        blue_channel = img[:, 2:3, :, :]

        # Apply different SRM filters to each channel
        red_out = F.conv2d(red_channel, self.kernel[0:1, :, :, :], stride=1, padding=2)
        green_out = F.conv2d(green_channel, self.kernel[0:1, :, :, :], stride=1, padding=2)
        blue_out = F.conv2d(blue_channel, self.kernel[0:1, :, :, :], stride=1, padding=2)

        # Concatenate the channels back together
        out = torch.cat((red_out, green_out, blue_out), dim=1)
        out = self.truc(out)

        # Remove batch dimension
        out = out.squeeze(0)
        
        # Convert back to PIL Image
        out = TF.to_pil_image(out)
        
        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, -2, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        filters = [
            # filter1,
            # filter2,
            filter3
            ]
        filters = np.array(filters)
        filters = filters[:, np.newaxis, :, :]  # Add channel dimension to filters
        filters = torch.FloatTensor(filters)    # (3,1,5,5)
        return filters

class AlbumentationsSRMConv2dOnlyFilter3(ImageOnlyTransform):
    def __init__(self, inc=3, learnable=False, always_apply=False, p=1.0):
        super(AlbumentationsSRMConv2dOnlyFilter3, self).__init__(always_apply, p)
        self.transform = SRMConv2dTransformOnlyFilter3(inc=inc, learnable=learnable)

    def apply(self, img, **params):
        img = Image.fromarray(img)
        img = self.transform(img)
        return np.array(img)


def get_transform_to_eval_filter3(size):
    return Compose([
        AlbumentationsSRMConv2dOnlyFilter3(inc=3, learnable=False),
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
        ToGray(p=0.2),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_transform_to_eval_NO_SRM(size):
    return Compose([
        # IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        # PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
        # ToGray(p=0.2),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_transform_to_eval_Sobel(size):
    return Compose([
            # AlbumentationsSRMConv2d(inc=3, learnable=False),
            SobelFilterTransform(p=1.0),
            # IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            # PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
            # ToGray(p=0.2),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])