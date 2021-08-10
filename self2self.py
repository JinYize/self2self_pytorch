###########################################################
# This script tests a modified lenet5.
###########################################################

import numpy as np # linear algebra

import os
import glob

from matplotlib import cm
from matplotlib import pyplot as plt
import cv2
import util
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import utils,models
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import trange
from time import sleep
from scipy.io import loadmat
import torchvision.datasets as dset
from torch.utils.data import sampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from partialconv2d import PartialConv2d
from model import self2self


def image_loader(image, device, p1, p2):
    """load image, returns cuda tensor"""
    loader = T.Compose([T.RandomHorizontalFlip(torch.round(torch.tensor(p1))),T.RandomVerticalFlip(torch.round(torch.tensor(p2))),T.ToTensor()])
    image = Image.fromarray(image.astype(np.uint8))
    image = loader(image).float()
    image = torch.tensor(image)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.to(device)

if __name__ == "__main__":
	##Enable GPU
	USE_GPU = True
	
	dtype = torch.float32
	
	if USE_GPU and torch.cuda.is_available():
	    device = torch.device('cuda')
	else:
	    device = torch.device('cpu')
	
	print('using device:', device)
	
	model = self2self(3,0.3)
	img = np.array(Image.open("5.png"))
	learning_rate = 1e-4
	model = model.cuda()
	optimizer = optim.Adam(model.parameters(), lr = learning_rate)
	w,h,c = img.shape
	p=0.3
	NPred=100
	slice_avg = torch.tensor([1,3,512,512]).to(device)
	for itr in range(500000):
	    p_mtx = np.random.uniform(size=[img.shape[0],img.shape[1],img.shape[2]])
	    mask = (p_mtx>p).astype(np.double)*0.7
	    #img_input = img*mask
	    img_input = img
	    #y = img - img_input
	    y = img
	    p1 = np.random.uniform(size=1)
	    p2 = np.random.uniform(size=1)
	    img_input_tensor = image_loader(img_input, device, p1, p2)
	    y = image_loader(y, device, p1, p2)
	    mask = np.expand_dims(np.transpose(mask,[2,0,1]),0)
	    mask = torch.tensor(mask).to(device, dtype=torch.float32)
	    
	    model.train()
	    img_input_tensor = img_input_tensor*mask
	    output = model(img_input_tensor, mask)
	    
	    loader = T.Compose([T.RandomHorizontalFlip(torch.round(torch.tensor(p1))),T.RandomVerticalFlip(torch.round(torch.tensor(p2)))])
	    if itr == 0:
	        slice_avg = loader(output)
	    else:
	        slice_avg = slice_avg*0.99+loader(output)*0.01
	    #output = model(torch.mul(img_input_tensor,mask))
	    #LossFunc = nn.MSELoss(reduction='sum')
	    #loss = LossFunc(output*(mask), y*(mask))/torch.sum(mask)
	    loss = torch.sum((output-y)*(output-y)*(1-mask))/torch.sum(1-mask)
	    optimizer.zero_grad()
	    loss.backward()
	    optimizer.step()
	    
	    print(torch.max(output), torch.max(y))
	    print("iteration %d, loss = %.4f" % (itr+1, loss.item()*100))
	    #break
	    if (itr+1)%1000 == 0:
	        model.eval()
	        sum_preds = np.zeros((img.shape[0],img.shape[1],img.shape[2]))
	        for j in range(NPred):
	            p_mtx = np.random.uniform(size=img.shape)
	            mask = (p_mtx>p).astype(np.double)
	            img_input = img*mask
	            img_input_tensor = image_loader(img_input, device, 0.1, 0.1)
	            mask = np.expand_dims(np.transpose(mask,[2,0,1]),0)
	            mask = torch.tensor(mask).to(device, dtype=torch.float32)
	            
	            output_test = model(img_input_tensor,mask)
	            sum_preds[:,:,:] += np.transpose(output_test.detach().cpu().numpy(),[2,3,1,0])[:,:,:,0]
	        avg_preds = np.squeeze(np.uint8(np.clip((sum_preds-np.min(sum_preds)) / (np.max(sum_preds)-np.min(sum_preds)), 0, 1) * 255))
	        #avg_preds = np.transpose(slice_avg.detach().cpu().numpy(),[2,3,1,0])[:,:,:,0]
	        #avg_preds = np.squeeze(np.uint8(np.clip((avg_preds-np.min(avg_preds)) / (np.max(avg_preds)-np.min(avg_preds)), 0, 1) * 255))
	        write_img = Image.fromarray(avg_preds)
	        write_img.save("images/Self2self-"+str(itr+1)+".png")
	        torch.save(model.state_dict(),'models/model-'+str(itr+1))
