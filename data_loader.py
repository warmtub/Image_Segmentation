import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from pydicom import dcmread
import json
from labelme import utils

class ImageFolder(data.Dataset):
    def __init__(self, root,image_size=224,mode='train',augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        self.root = root
        
        # GT : Ground Truth
        self.GT_paths = root[:-1]+'_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_paths = [path for path in self.image_paths if path.split('.')[-1] == 'dcm']
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0,90,180,270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode,len(self.image_paths)))
        
        self.label_name_to_value = {}
        with open(os.path.join(self.root, "classes.txt")) as f:
            value = 0
            for line in f:
                self.label_name_to_value[line[:-1]] = value
                value += 1

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        #print(f'image_path: {image_path}')
        #filename = image_path.split('_')[-1][:-len(".jpg")]
        
        #Windowing (CT)
        #https://radiopaedia.org/articles/windowing-ct
        ds = dcmread(image_path)
        image = ds.pixel_array
        h, w = image.shape
        image = image.astype('float64')
        intercept = ds.RescaleIntercept
        try:
            wc = ds.WindowCenter[0]
            ww = ds.WindowWidth[0]
            UL = wc + ww/2
            LL = wc - ww/2
        except:
            wc = ds.WindowCenter
            ww = ds.WindowWidth
            UL = wc + ww/2
            LL = wc - ww/2
        slope = ds.RescaleSlope
        image -= (-intercept+LL)
        image[image<0] = 0
        image[image>(UL-LL)] = UL-LL
        image *= 255.0/image.max()
        image = image.astype('uint8')
        image = Image.fromarray(image)
        image = image.convert('RGB')
        
        #aspect_ratio = image.size[1]/image.size[0]
        Transform = []
        #ResizeRange = random.randint(300,320)
        #Transform.append(T.Resize((int(ResizeRange*aspect_ratio),ResizeRange)))
        #p_transform = random.random()

        
        """
        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            #print('augmentation')
            RotationDegree = random.randint(0,3)
            RotationDegree = self.RotationDegree[RotationDegree]
            if (RotationDegree == 90) or (RotationDegree == 270):
                aspect_ratio = 1/aspect_ratio

            Transform.append(T.RandomRotation((RotationDegree,RotationDegree)))
                        
            RotationRange = random.randint(-10,10)
            Transform.append(T.RandomRotation((RotationRange,RotationRange)))
            CropRange = random.randint(250,270)
            Transform.append(T.CenterCrop((int(CropRange*aspect_ratio),CropRange)))
            Transform = T.Compose(Transform)
            
            image = Transform(image)
            GT = Transform(GT)

            ShiftRange_left = random.randint(0,20)
            ShiftRange_upper = random.randint(0,20)
            ShiftRange_right = image.size[0] - random.randint(0,20)
            ShiftRange_lower = image.size[1] - random.randint(0,20)
            image = image.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))
            GT = GT.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))

            if random.random() < 0.5:
                image = F.hflip(image)
                GT = F.hflip(GT)

            if random.random() < 0.5:
                image = F.vflip(image)
                GT = F.vflip(GT)

            Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)

            image = Transform(image)

            Transform =[]
        """

        #Transform.append(T.Resize((256,256)))
        Transform.append(T.ToTensor())
        Transform.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        Transform = T.Compose(Transform)
        image = Transform(image)

        GT_path = image_path.split('.dcm')[0] + '_1.json'
        #GT = Image.open(GT_path)
        json_data = json.load(open(GT_path))
        gt_label = utils.shapes_to_label(ds.pixel_array.shape, json_data['shapes'], self.label_name_to_value)
        #gt_img = utils.draw_label(gt_label, ds.pixel_array)
        
        #print('before', np.unique(gt_label), gt_label.shape)
        gt_label = torch.tensor(np.array(gt_label), dtype=torch.int64)
        gt_label = torch.nn.functional.one_hot(gt_label, 12).to(torch.float).permute(2,0,1)
        #print('after', np.unique(gt_label), gt_label.shape)

        return image, gt_label

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
    """Builds and returns Dataloader."""
    
    dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader
