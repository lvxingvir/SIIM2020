import numpy as np
import pandas as pd
import os
import cv2
import torch.nn.init as init
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim import Adam, SGD
import time
from time import gmtime, strftime
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch.functional as F
from tqdm import tqdm
from sklearn import metrics
import urllib
import pickle
import cv2
import torch.nn.functional as F

#%%

from torchvision import models
import seaborn as sns
import random

import warnings
warnings.filterwarnings(action='ignore')

#%%

from sklearn.metrics import roc_auc_score
import sys
# sys.path.append('./pytorch-auto-augment')

from auto_augment.auto_augment import AutoAugment, Cutout
from EfficientUnet.efficientunet import *
from sampler import BalancedBatchSampler
from tools import add_image
from lr_cosine import CosineAnnealingWarmUpRestarts
from Attention_Resnet.model.residual_attention_network_mm import ResidualAttentionModel_92 as resattnet

import imgaug as ia
from imgaug import augmenters as iaa

train_path = r'C:\Users\Xing\Projects\SIIM2020\data_224\train'
test_path = r'C:\Users\Xing\Projects\SIIM2020\data_224\test'
train_csv = pd.read_csv(r'C:\Users\Xing\Projects\SIIM2020\data\train.csv')
test_csv = pd.read_csv(r'C:\Users\Xing\Projects\SIIM2020\data\test.csv')
sample = pd.read_csv(r'C:\Users\Xing\Projects\SIIM2020\data\sample_submission.csv')




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def img_augmentation(img,seq_det):

    img = img.transpose(2,0,1).astype(np.float32)

    for i in range(len(img)):
        img[i,:,:] = seq_det.augment_images(img[i,:,:])

    img = img.transpose(1,2,0).astype(np.float64)

    return img

class MyDataset(Dataset):

    def __init__(self, dataframe, transform=None, test=False):
        self.df = dataframe
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        label = self.df.target.values[idx]
        p = self.df.image_name.values[idx]

        if self.test == False:
            # p_path = train_path + p + '.png'
            p_path = os.path.join(train_path,p+'.png')
        else:
            p_path = os.path.join(test_path,p+'.png')

        image = cv2.imread(p_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #         image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)


        return image, label

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_model(model,epoch,optimizer,train_loader,criterion):
    model.train()

    losses = AverageMeter()
    avg_loss = 0.

    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
    for idx, (imgs, labels) in enumerate(train_loader):
        imgs_train, labels_train = imgs.cuda(), labels.cuda()
        #imgs_train, labels_train = imgs.cuda(), labels.float().cuda()
        output_train = model(imgs_train)
        labels_train = torch.unsqueeze(labels_train,1).float()
        loss = criterion(output_train, labels_train)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        avg_loss += loss.item() / len(train_loader)

        losses.update(loss.item(), imgs_train.size(0))

        tk.set_postfix(loss=losses.avg)
        tk.update(1)

        if idx % 5000 == 0:
            add_image(imgs, output_train, labels_train, writer, subset='train', epoch=epoch, name= str(idx)+'_image')

    return avg_loss


def test_model(model,epoch,val_loader,criterion):
    model.eval()

    losses = AverageMeter()
    avg_val_loss = 0.

    valid_preds, valid_targets = [], []

    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for idx, (imgs, labels) in enumerate(val_loader):
            imgs_valid, labels_valid = imgs.cuda(), labels.cuda().long()
            output_valid = model(imgs_valid)
            labels_valid = torch.unsqueeze(labels_valid, 1).float()
            loss = criterion(output_valid, labels_valid)

            avg_val_loss += loss.item() / len(val_loader)

            losses.update(loss.item(), imgs_valid.size(0))

            tk.set_postfix(loss=losses.avg)
            tk.update(1)

            if idx % 1000 == 0:
                add_image(imgs, output_valid, labels_valid, writer, subset='valid', epoch=epoch,
                          name=str(idx) + '_image')

            valid_preds.append(torch.softmax(output_valid, 1)[:, 1].detach().cpu().numpy())
            valid_targets.append(labels_valid.detach().cpu().numpy())

        valid_preds = np.concatenate(valid_preds)
        valid_targets = np.concatenate(valid_targets)
        auc = roc_auc_score(valid_targets, valid_preds)

    return avg_val_loss, auc

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'

    seed_everything(2020)
    num_classes = 2
    bs = 32
    lr = 1e-3
    IMG_SIZE = 224

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testset = MyDataset(sample, transform=test_transform, test=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=4)

    kf = StratifiedKFold(5, shuffle=True, random_state=0)

    cv = []

    fold = 0

    for trn_ind, val_ind in kf.split(train_csv.image_name, train_csv.target):
        fold += 1
        print('fold:', fold)

        ## logs
        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        # time_string = strftime("%a%d%b%Y", gmtime())
        result_path = r'C:\Users\Xing\Projects\SIIM2020\train_log'
        descrip = 'July19_attenres_bce_longtrain_fold_'+ str(fold)
        model_save_path = os.path.join(result_path, descrip, time_string, 'save')
        tb_save_path = os.path.join(result_path, descrip, time_string, 'tb')
        roc_save_path = os.path.join(result_path, descrip, time_string, 'roc')
        troc_save_path = os.path.join(result_path, descrip, time_string, 'troc')
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
            os.makedirs(tb_save_path)
            os.makedirs(roc_save_path)
            os.makedirs(troc_save_path)
        writer = SummaryWriter(logdir=tb_save_path)

        train_df = train_csv.loc[trn_ind]
        val_df = train_csv.loc[val_ind]
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        trainset = MyDataset(train_df, transform=train_transform)
        # train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=4)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                                   sampler=BalancedBatchSampler(trainset, type='single_label'),num_workers=2)

        valset = MyDataset(val_df, transform=test_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=bs, shuffle=False, num_workers=4)

        #     model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True, num_classes=num_classes)
        # model = EfficientNet.from_name('efficientnet-b3', n_classes=2, pretrained=False).cuda()
        model = resattnet().cuda()
        #     model.cuda()


        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-8, weight_decay=0.001)
        # criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCELoss()
        criterion = nn.BCEWithLogitsLoss()
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=8, T_mult=2, eta_max=1e-3, T_up=1, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, threshold=0.05, factor=0.5)

        Train_C_flag = False
        best_auc = 0
        n_epochs = 80
        es = 0

        if Train_C_flag == True:
            model_load_path = r'C:\Users\Xing\Projects\SIIM2020\train_log\Jun12_effnetb3_fold_1\Sat13Jun2020-004141\save'
            model_name = r'\best_model_auc.pth'

            # model_load_path = r'pretrain'
            # model_name = r'\resnet_34.pth'

            checkpoint = torch.load(model_load_path + model_name)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            Epoch = checkpoint['epoch']
        else:
            Epoch = 0

        for epoch in range(Epoch,Epoch+n_epochs):
            avg_loss = train_model(model, epoch+1,optimizer,train_loader,criterion)
            avg_val_loss, auc = test_model(model,epoch+1,val_loader,criterion)

            if auc > best_auc:
                best_auc = auc
                # torch.save(model.state_dict(), str(fold) + 'weight.pt')
                save_file_path = os.path.join(model_save_path, 'best_model_auc.pth')
                states = {'epoch': epoch +1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(states, save_file_path)
            else:
                es += 1
                if es > 5:
                    pass

            print('\n','fold:',fold,'epoch:', epoch+1,'current_val_auc:', auc, 'best_val_auc:', best_auc, 'avg_val_loss:', avg_val_loss)

            scheduler.step()
            # scheduler.step(avg_loss)


            writer.add_scalars('loss/epoch',
                               {'train loss': avg_loss, 'validation loss': avg_val_loss}, epoch + 1)
            # writer.add_scalars('acc/epoch',
            #                    {'train accuracy': accuracies.avg, 'validation accuracy': accuracies_val.avg}, epoch + 1)
            writer.add_scalars('Learning Rate/epoch',
                               {'train accuracy': optimizer.param_groups[0]['lr']}, epoch + 1)
            writer.add_scalars('roc_auc/epoch', {'val_auc': auc}, epoch + 1)

            print('\n wait for a second....\n')

            torch.cuda.empty_cache()

            # time.sleep(100)
            wk = tqdm(range(1))
            for i in wk:
                time.sleep(150)




        cv.append(best_auc)
        print(cv)