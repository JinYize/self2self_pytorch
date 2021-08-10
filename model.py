import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import utils,models
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL
from tqdm import trange
from time import sleep
from scipy.io import loadmat
import torchvision.datasets as dset
from torch.utils.data import sampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from partialconv2d import PartialConv2d

class EncodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, flag):
        super(EncodeBlock, self).__init__()
        self.conv = PartialConv2d(in_channel, out_channel, kernel_size = 3, padding = 1)
        #self.conv = nn.Conv2d(in_channel, out_channel, kernel_size = 3, padding = 1)
        self.nonlinear = nn.LeakyReLU(0.1)
        self.MaxPool = nn.MaxPool2d(2)
        self.flag = flag
        
    def forward(self, x, mask_in):
        out1, mask_out = self.conv(x, mask_in = mask_in)
        out2 = self.nonlinear(out1)
        if self.flag:
            out = self.MaxPool(out2)
            mask_out = self.MaxPool(mask_out)
        else:
            out = out2
        return out, mask_out


class DecodeBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, final_channel = 3, p = 0.7, flag = False):
        super(DecodeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(out_channel, final_channel, kernel_size = 3, padding = 1)
        self.nonlinear1 = nn.LeakyReLU(0.1)
        self.nonlinear2 = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        self.flag = flag
        self.Dropout = nn.Dropout(p)
        
    def forward(self, x):
        out1 = self.conv1(self.Dropout(x))
        out2 = self.nonlinear1(out1)
        out3 = self.conv2(self.Dropout(out2))
        out4 = self.nonlinear2(out3)
        if self.flag:
            out5 = self.conv3(self.Dropout(out4))
            out = self.sigmoid(out5)
        else:
            out = out4
        return out

class self2self(nn.Module):
    def __init__(self, in_channel, p):
        super(self2self, self).__init__()
        self.EB0 = EncodeBlock(in_channel, 48, flag=False)
        self.EB1 = EncodeBlock(48, 48, flag=True)
        self.EB2 = EncodeBlock(48, 48, flag=True)
        self.EB3 = EncodeBlock(48, 48, flag=True)
        self.EB4 = EncodeBlock(48, 48, flag=True)
        self.EB5 = EncodeBlock(48, 48, flag=True)
        self.EB6 = EncodeBlock(48, 48, flag=False)
        
        self.DB1 = DecodeBlock(96, 96, 96,p=p)
        self.DB2 = DecodeBlock(144, 96, 96,p=p)
        self.DB3 = DecodeBlock(144, 96, 96,p=p)
        self.DB4 = DecodeBlock(144, 96, 96,p=p)
        self.DB5 = DecodeBlock(96+in_channel, 64, 32, in_channel,p=p, flag=True)
        
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.concat_dim = 1
    def forward(self, x, mask):
        out_EB0, mask = self.EB0(x, mask)
        out_EB1, mask = self.EB1(out_EB0, mask)
        out_EB2, mask = self.EB2(out_EB1, mask_in = mask)
        out_EB3, mask = self.EB3(out_EB2, mask_in = mask)
        out_EB4, mask = self.EB4(out_EB3, mask_in = mask)
        out_EB5, mask = self.EB5(out_EB4, mask_in = mask)
        out_EB6, mask = self.EB6(out_EB5, mask_in = mask)
        
        out_EB6_up = self.Upsample(out_EB6)
        in_DB1 = torch.cat((out_EB6_up, out_EB4),self.concat_dim)
        out_DB1 = self.DB1((in_DB1))
        
        out_DB1_up = self.Upsample(out_DB1)
        in_DB2 = torch.cat((out_DB1_up, out_EB3),self.concat_dim)
        out_DB2 = self.DB2((in_DB2))
        
        out_DB2_up = self.Upsample(out_DB2)
        in_DB3 = torch.cat((out_DB2_up, out_EB2),self.concat_dim)
        out_DB3 = self.DB3((in_DB3))
        
        out_DB3_up = self.Upsample(out_DB3)
        in_DB4 = torch.cat((out_DB3_up, out_EB1),self.concat_dim)
        out_DB4 = self.DB4((in_DB4))
        
        out_DB4_up = self.Upsample(out_DB4)
        in_DB5 = torch.cat((out_DB4_up, x),self.concat_dim)
        out_DB5 = self.DB5(in_DB5)
        
        return out_DB5
