import torchvision.datasets.voc as voc
from pycocotools.coco import COCO
from torchvision import datasets as datasets
import torch
from PIL import Image
import os
from PIL import ImageDraw
import numpy as np
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from randaugment import RandAugment
from util import encode_labels
from nuswide import nuswide

class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        print(target.shape)
        return img, target

class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


class PascalVOC_Dataset(voc.VOCDetection):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.
        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years 2007 to 2012.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
                (default: alphabetic indexing of VOC's 20 classes).
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, required): A function/transform that takes in the
                target and transforms it.
    """
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):

        super().__init__(
             root,
             year=year,
             image_set=image_set,
             download=download,
             transform=transform,
             target_transform=target_transform)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        return super().__getitem__(index)


    def __len__(self):
        """
        Returns:
            size of the dataset
        """
        return len(self.images)


def create_dataset(args):

    data_dir = args.data + args.dataset
    batch_size = args.batch_size
    download_data = args.download_data

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    if args.dataset =='COCO2014':
        # COCO DataLoader
        instances_path_val = os.path.join(data_dir, 'annotations/instances_val2014.json')
        instances_path_train = os.path.join(data_dir, 'annotations/instances_train2014.json')
        data_path_val   = f'{data_dir}/val2014'    # args.data
        data_path_train = f'{data_dir}/train2014'  # args.data
        COCO_val_dataset = CocoDetection(data_path_val,
                                    instances_path_val,
                                    transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std),
                                        # normalize, # no need, toTensor does normalization
                                    ]))
        COCO_train_dataset = CocoDetection(data_path_train,
                                        instances_path_train,
                                        transforms.Compose([
                                            transforms.Resize((args.image_size, args.image_size)),
                                            CutoutPIL(cutout_factor=0.5),
                                            # RandAugment(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std),
                                            # normalize,
                                        ]))
        print('Using COCO dataset')
        print("COCO len(val_dataset)): ", len(COCO_val_dataset))
        print("COCO len(train_dataset)): ", len(COCO_train_dataset))
        train_loader = torch.utils.data.DataLoader(
            COCO_train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)

        valid_loader = torch.utils.data.DataLoader(
            COCO_val_dataset, batch_size=args.batch_size,
            num_workers=args.workers, drop_last=True)
    elif args.dataset == 'VOC2012':
        # Create VOC train dataloader
        transformations = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                            transforms.RandomChoice([
                                                transforms.ColorJitter(brightness=(0.80, 1.20)),
                                                transforms.RandomGrayscale(p = 0.25)
                                                ]),
                                            transforms.RandomHorizontalFlip(p = 0.25),
                                            transforms.RandomRotation(25),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = mean, std = std),
                                            ])

        VOC_dataset_train = PascalVOC_Dataset(data_dir,
                                            year='2012',
                                            image_set='train',
                                            download=download_data,
                                            transform=transformations,
                                            target_transform=encode_labels)

        # VOC validation dataloader
        transformations_valid = transforms.Compose([transforms.Resize(args.image_size),
                                                transforms.CenterCrop(args.image_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean = mean, std = std),
                                                ])

        VOC_dataset_valid = PascalVOC_Dataset(data_dir,
                                            year='2012',
                                            image_set='val',
                                            download=download_data,
                                            transform=transformations_valid,
                                            target_transform=encode_labels)
        train_loader = DataLoader(VOC_dataset_train, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(VOC_dataset_valid, batch_size=batch_size, num_workers=4, drop_last=True)

        # VOC testing loader
        transformations_test = transforms.Compose([transforms.Resize(args.image_size),
                                                transforms.FiveCrop(args.image_size),
                                                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                                transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean = mean, std = std)(crop) for crop in crops])),
                                                ])

        dataset_test = PascalVOC_Dataset(data_dir,
                                            year='2012',
                                            image_set='val',
                                            download=download_data,
                                            transform=transformations_test,
                                            target_transform=encode_labels)
        
    elif args.dataset == 'NUSWIDE':
        transformations_valid = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean = mean, std = std),
                                                    ])
        
        vaild_dataset = nuswide.NUSWIDEClassification(root='./nuswide',
                                                      set='test',
                                                      transform=transformations_valid)
        
        transformations_train = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomResizedCrop((args.image_size, args.image_size), scale=(0.7, 1.0)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean = mean, std = std),
                                                    ])
        
        train_dataset = nuswide.NUSWIDEClassification(root='./nuswide',
                                                      set='trainval',
                                                      transform=transformations_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(vaild_dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    return train_loader, valid_loader