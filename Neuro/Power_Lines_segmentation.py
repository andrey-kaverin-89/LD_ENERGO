#!/usr/bin/env python
# coding: utf-8


import torchvision
import glob
import os
import cv2
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.utils.data as data
import albumentations as A
get_ipython().run_line_magic('matplotlib', 'inline')

## Путь к датасету, имена папок с инпутом и таргетом
DATASET_PATH = './Dataset/'
MASK_DIR = 'pl_mask'
IMAGE_DIR = 'im'



## Кастомный класс торч датасета, позволяющий делать одинаковые
## аугментации на инпут и таргет
class DatasetSegmentation(data.Dataset):
    def __init__(self, folder_path, image_dir, mask_dir, val = False):
        super(DatasetSegmentation, self).__init__()
        self.val = val
        
        if val ==True:
            self.img_files = glob.glob(os.path.join(folder_path,'val',image_dir,'*.bmp'))
            self.mask_files = glob.glob(os.path.join(folder_path,'val',mask_dir,'*.bmp'))
        else:
            self.img_files = glob.glob(os.path.join(folder_path,'train',image_dir,'*.bmp'))
            self.mask_files = glob.glob(os.path.join(folder_path,'train',mask_dir,'*.bmp'))

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            
            data = cv2.imread(img_path)
            label = cv2.imread(mask_path)
            
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            
            if self.val ==False:
                transformed = self.transform_tr(image = data, mask = label)
                data, label = transformed['image'], transformed['mask']
            else:
                transformed = self.transform_val(image = data, mask = label)
                data, label = transformed['image'], transformed['mask']
                
                
            label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
            label[label>1] = 1
            
            return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.img_files)
    
    ## Аугментации на обучающую выборку
    def transform_tr(self, image, mask):
        transformation = A.Compose([
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5)])
        return transformation(image=image, mask=mask)

    ## Преобразование на валидационную выборку
    def transform_val(self, image, mask):
        transformation = A.Compose([
                    A.CenterCrop(width=256, height=256)])
        return transformation(image=image, mask=mask)
    
    
    
## Функция обучения модели
def train(dataset_path = DATASET_PATH, image_dir = IMAGE_DIR, mask_dir = MASK_DIR):
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    loss = nn.BCELoss()
    
    ## Подгрузка архитектуры U-Net
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)
    
    ## Фриз слоев для файнтюна
    for param in model.parameters():
        param.requires_grad = False
        
    ##Добавление сигмоиды, которой не хватает в архитектуре    
    model.conv = nn.Sequential(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1)), nn.Sigmoid())
    
    lr = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    n_epoches = 20
    
    train_dataset = DatasetSegmentation(dataset_path, image_dir, mask_dir)
    train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=2, shuffle=True, num_workers=0)
    
    val_dataset = DatasetSegmentation(dataset_path, image_dir, mask_dir, val = True)
    val_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=2, shuffle=False, num_workers=0)
    

    for i in range(n_epoches):
        model.train()

        for x_train, y_train in train_dataloader:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_train = torch.unsqueeze(y_train, dim = -1)
            x_train = x_train.permute((0,3,1,2))
            y_train = y_train.permute((0,3,1,2))

            optimizer.zero_grad()

            y_pred = model.forward(x_train)
            ce = loss(y_pred, y_train)

            ce.backward()
            optimizer.step()

        hist = []    
        for x_val, y_val in val_dataloader:

            model.eval()

            x_val = x_val.to(device)
            y_val = y_val.to(device)
            y_val = torch.unsqueeze(y_val, dim = -1)
            x_val = x_val.permute((0,3,1,2))
            y_val = y_val.permute((0,3,1,2))

            y_pred = model.forward(x_val)
            ce_val = loss(y_pred, y_val)

            hist.append(ce_val)

        val_loss = sum(hist)/len(hist)
        
    torch.save(model.state_dict(), './chp/model_pl.pt') 
    
## Инференс модели    
def inference(x_path, save_path = './inference', model_path = './chp/model_pl.pt'):

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=False)
    model.conv = nn.Sequential(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1)), nn.Sigmoid())
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    t = cv2.imread(x_path)
    transformation = A.Compose([A.CenterCrop(width=256, height=256)])
    x = transformation(image=t)['image']
    x = torch.from_numpy(x).float()
    x = torch.unsqueeze(x, dim = 0)
    x = x.permute((0,3,1,2))


    pred_mask = model.forward(x)
    
    ##Преобразование маски
    pred_mask = pred_mask[0][0]
    pred_mask = pred_mask.detach().numpy()
    pred_mask[pred_mask>=0.65] = 1
    pred_mask[pred_mask<0.65] = 0
    pred_mask = pred_mask*256
    pred_mask = cv2.cvtColor(pred_mask,cv2.COLOR_GRAY2RGB)
    pred_mask = pred_mask.transpose(2,0,1)
    pred_mask[0] = 0
    pred_mask[1] = 0
    pred_mask = pred_mask.transpose(1,2,0).astype(int)
    
    cv2.imwrite(save_path+ '/' + x_path.split('/')[-1] + '_segmentation.bmp', pred_mask)
