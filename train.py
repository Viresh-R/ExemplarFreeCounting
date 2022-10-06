#!/usr/bin/env python
# coding: utf-8

# In[21]:


import json
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from collections import OrderedDict
from torchvision.ops import box_iou
from model import CountRegressorNoInterpolationBPoolINLeakySkipPool
from model import CountRegressorNoInterpolationBPoolINLeaky
from random import randint
orig_path = '/nfs/bigneuron.cs.stonybrook.edu/viresh/class-agnostic-new/ClassAgnosticCounting/FSC148'
sys.path.insert(0, orig_path)
from datasetBoxesBatchWCropDotsPastebigneuronCreateExemplars import CountingDataset, Normalize, ResizeExampleImage, ResizeImage, ToTensor,CropImage
from core.model import weights_normal_init,ResNet50Conv
#from model import CountRegressorNoInterpolationBPool as CountRegressor
#from core.model import CountRegressor2X as CountRegressor 
#from model import CountRegressorNoInterpolationBPool as CountRegressor
from os.path import basename, exists, join, splitext
import math
from torch.utils.data import DataLoader
from torchvision import transforms
from utilsGAN import  format_for_plotting,denormalize,visualizeOutputQuery,standardize_and_clip,getGaussianFromDot
from utilsA import extract_features,extract_featuresManifoldMixup
#from models import GeneratorUNet,DiscriminatorUNet
#from networksCycleGAN import define_G, define_D
#from model import FeatMapINLeaky as FeatMap
#from networksCycleGAN import FeatMapINLeaky as FeatMap
import copy
import torchvision
from torch.utils.tensorboard import SummaryWriter
from random import randint
from utilsGAN import getGaussianFromDotA,visualizeOutputQuery,closest_dist,apply_gaussian_mask
from utilsDemo import pad_to_size
from utilsA import get_Mask 
import shutil
from utilsA import visualizeExemplar

torch.backends.cudnn.benchmark = True
from torchvision.ops import box_iou


# In[3]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
#devices_ids = [1,2,3]
#device_ids = [1,2,3]
# In[3]:
base_dir = "/nfs/bigneuron.cs.stonybrook.edu/viresh/FSC_NewDataOnly" # Dataset Name

# In[4]:
dataset_dir = join(base_dir, "dataset.json")
json_annotation = "/nfs/bigneuron.cs.stonybrook.edu/viresh/FSC_NewDataOnly/json_annotationCombined_384_VarV2_updated1.json"
images_dir = join(base_dir, "images_384_VarV2")
density_map_dir = join(base_dir, "gt_density_map_adaptive_384_VarV2")
#src_dot_dir = '/nfs/bigneuron/viresh/class-agnostic-new/ClassAgnosticCounting/GAN/JitteredDataset384/dots_original'
src_dot_dir = '/nfs/bigcornea/add_disk0/viresh/data/JitteredDataset384/dots_original'


if not exists('save'): os.makedirs('save')
if not exists('logs'): os.makedirs('logs')


RANDOM_SEED = 600
MAX_EPOCHS = 1500
BATCH_SIZE = 1
CROP_SIZE = 384
NUM_EXAMPLES = 3
EXAMPLE_TYPE = "box"
LEARNING_RATE = 1e-5
MAX_HW = 1504
weightMap = 1e-14
#DATA_SPLIT = class_exclusive_data_split
MAPS = ['map3','map4'] # '0','1','2','3' are 4, 8, 16, 32 times smaller respectively
Scales = [0.9, 1.1]
PreTrained = True

MODEL_NAME_PREFIX = "ZSC_Top3Proposals_DenseRPN_KD_FamNetB_BinarizedKT_RPN_MeanPooledC_SortedCount_RPNMapA"
outdir = '/nfs/bigbox.cs.stonybrook.edu/viresh/projects/FSC/ZSC/logs/' + MODEL_NAME_PREFIX
#outdir = '/nfs/biglens.cs.stonybrook.edu/viresh/logs/' + MODEL_NAME_PREFIX
#outdir = '/nfs/bigcornea/add_disk0/viresh/data/Result/' + MODEL_NAME_PREFIX
if not exists(outdir): os.makedirs(outdir)
wpixel = 1e-5    
weightG = 1e-5
weightD= 1e-5 
wCountReal = 1.
wCountFake = 10.
num_workers = 1
DISP_STEP = 50
resume_param = False
iou_thresh = 0.1
import shutil
import pickle


# In[23]:


#writer.close()
import pickle5
#proposal_filename = '/nfs/bigbox.cs.stonybrook.edu/viresh/projects/FSC/ZSC/data/Proposals_Top3_NearestNeighbor_FromTop50.pkl'
#proposal_filename = '/nfs/bigbox.cs.stonybrook.edu/viresh/projects/FSC/ZSC/data/Proposals_Top3_ClusteringAlgo1_Support3_50.pkl'
#proposal_filename = '/nfs/bigbox.cs.stonybrook.edu/viresh/projects/FSC/ZSC/data/Proposals_Top3_DenseRPN_352_704.pkl'
proposal_filename = '/nfs/bigbox.cs.stonybrook.edu/viresh/projects/FSC/ZSC/data/Proposals_Top3_DenseRPN_384_768_BinarizedKT_COCOFinetune.pkl'
proposal_filename = '/nfs/bigbox.cs.stonybrook.edu/viresh/projects/FSC/ZSC/data/Proposals_Top3_DenseRPN_384_768_BinarizedKT_COCOFinetune_ZSC_TransformerCounterC_COCO_RPN_FineTune_DualRPN_BinarizedKTB_Top50_Save2.pkl'
proposal_filename = '/nfs/bigbox/viresh/projects/FSC/ZSC/data/Proposals_Top3_DenseRPN_384_768_BinarizedKT_COCOFinetune_ZSC_TransformerCounterC_COCO_RPN_FineTune_DualRPN_BinarizedKTB_Top50_WithCount_Save2.pkl'
with open(proposal_filename, 'rb') as handle:
    All_Proposal = pickle5.load(handle)
All_Proposal_new = dict()
All_Counts_new = dict()
for pathX in All_Proposal.keys():


    #image, boxes = sample_batched['image'].cuda(), sample_batched['boxes'].cuda()
    #pathX = sample_batched['path'][0]

    #boxes = All_Proposal[pathX][0:NUM_EXAMPLES][0].cuda().unsqueeze(0).unsqueeze(0)
    #counts = All_Proposal[pathX][0:NUM_EXAMPLES][1].cuda().unsqueeze(0).unsqueeze(0)
    boxes = All_Proposal[pathX][0][0:10].cuda()
    counts = All_Proposal[pathX][1][0:10].cuda()
    values,indices = torch.sort(counts,descending=True)  
    All_Proposal_new[pathX] = boxes[indices[0:NUM_EXAMPLES]]
    All_Counts_new[pathX] = counts[indices[0:NUM_EXAMPLES]]
    
proposal_filename = '/nfs/bigbox.cs.stonybrook.edu/viresh/projects/FSC/ZSC/data/AllBoxes_fromPoints.pkl'
with open(proposal_filename, 'rb') as handle:
    All_Proposal_Boxes = pickle.load(handle)
#shutil.rmtree(outdir)
#os.mkdir(outdir)
#writer = SummaryWriter(outdir)


disp_step = 500
train_epoch_loss = 0
train_mae = 0
train_rmse = 0
wPixel = 1e-4
wFeatureMatching = 1e-8
wCountGAN = 1e1
nstylelayers = 8
wRealCount = 1e3
weightSparsity = 1e6
wConsistency = .01
wCenter = 1e2
wGen = 1e-2
wGenCount = 1e2
wDis = 1e-2
wDisCount = 1e2
input_nc = 4
input_nc_noise = 128
output_nc = 3
output_nc_mask = 1
ngf = 64
ndf = 64
LEARNING_RATE = 1e-5
LEARNING_RATE2 = 1e-5
Alpha = 1.0


# In[25]:


class Resnet50FPN(nn.Module):
    def __init__(self):
        super(Resnet50FPN, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        children = list(self.resnet.children())
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]
        #self.conv5 = children[7]
    def forward(self, im_data):
        feat = OrderedDict()
        feat_map1 = self.conv1(im_data)
        feat_map1 = self.conv2(feat_map1)
        feat_map3 = self.conv3(feat_map1)
        feat_map4 = self.conv4(feat_map3)
        #feat_map5 = self.conv5(feat_map4)
        #feat['map1'] = feat_map1
        #feat['map2'] = feat_map2
        feat['map3'] = feat_map3
        feat['map4'] = feat_map4
        #feat['map5'] = feat_map5
        return feat


resnet50_conv = Resnet50FPN()

resnet50_conv = nn.DataParallel(resnet50_conv)
resnet50_conv.cuda()
resnet50_conv.eval()


regressor = CountRegressorNoInterpolationBPoolINLeakySkipPool(7,pool='mean')

regressor.cuda()

teacher_FamNet = CountRegressorNoInterpolationBPoolINLeaky(6)

teacher_FamNet.cuda()

teacher_path = '/nfs/bigneuron.cs.stonybrook.edu/viresh/GithubVR/v05/LearningToCountEverything/data/pretrainedModels/FamNet_Save1.pth'
teacher_path = '/nfs/bigcornea.cs.stonybrook.edu/add_disk0/viresh/data/Result/FSC148_with_3_examples_FSC147_DualDisFamNet_NonGAN_Baseline_Leaky_IN/FSC148_with_3_examples_FSC147_DualDisFamNet_NonGAN_Baseline_Leaky_IN_1188.pth'

teacher_FamNet.load_state_dict(torch.load(teacher_path))
teacher_FamNet.eval()

PreTrained = False
if PreTrained == True:
    #path = '/nfs/bigneuron.cs.stonybrook.edu/viresh/GithubVR/FewShotDemo/logs/FamNet_922.pth'
    #path = '/nfs/bigneuron.cs.stonybrook.edu/viresh/GithubVR/LearningToCountEverything/data/pretrainedModels/FamNet_Save1.pth'
    path = '/nfs/bigcornea/add_disk0/viresh/data/Result/FSC148_with_3_examples_FSC147_DualDisFamNet_NonGAN_Baseline_Leaky_IN/FSC148_with_3_examples_FSC147_DualDisFamNet_NonGAN_Baseline_Leaky_IN_1188.pth'
    regressor.load_state_dict(torch.load(path))
if PreTrained == False:
    weights_normal_init(regressor, dev=0.001)
regressor = nn.DataParallel(regressor)
optimizer = optim.Adam(regressor.parameters(), lr = LEARNING_RATE)


# In[6]:


loss = nn.MSELoss().cuda()
criterion_GAN = torch.nn.MSELoss().cuda()
criterion_pixelwise = torch.nn.L1Loss().cuda()


# In[ ]:


all_fname = '/nfs/bigneuron/viresh/FSC_NewDataOnly/Image_LabelsV1.txt'
train_fname = '/nfs/bigneuron/viresh/FSC_NewDataOnly/Train_Classes.txt'
val_fname = '/nfs/bigneuron/viresh/FSC_NewDataOnly/Val_Classes.txt'
test_fname = '/nfs/bigneuron/viresh/FSC_NewDataOnly/Test_Classes.txt'
gray_fname = '/nfs/bigneuron/viresh/FSC_NewDataOnly/GrayImages.txt'
not_three = '/nfs/bigneuron/viresh/FSC_NewDataOnly/NotThreeBb.txt'
all_fname = '/nfs/bigneuron.cs.stonybrook.edu/viresh/FSC_NewDataOnly/Image_LabelsV1.txt'
train_fname = '/nfs/bigneuron.cs.stonybrook.edu/viresh/FSC_NewDataOnly/Train_Classes.txt'
val_fname = '/nfs/bigneuron.cs.stonybrook.edu/viresh/FSC_NewDataOnly/Val_Classes.txt'
test_fname = '/nfs/bigneuron.cs.stonybrook.edu/viresh/FSC_NewDataOnly/Test_Classes.txt'
gray_fname = '/nfs/bigneuron.cs.stonybrook.edu/viresh/FSC_NewDataOnly/GrayImages.txt'
not_three = '/nfs/bigneuron.cs.stonybrook.edu/viresh/FSC_NewDataOnly/NotThreeBb.txt'
lines_gray = [line.rstrip('\n') for line in open(gray_fname)]
lines_train = [line.rstrip('\n') for line in open(train_fname)]
lines_all = [line.rstrip('\n') for line in open(all_fname)]
lines_val = [line.rstrip('\n') for line in open(val_fname)]
lines_test = [line.rstrip('\n') for line in open(test_fname)]
train_classes = list()
val_classes = list()
test_classes  = list()
train_images = list()
test_images = list()
val_images = list()
for x in lines_train:
    x = x.split('\t')[0]
    train_classes.append(x)
for x in lines_val:
    x = x.split('\t')[0]
    val_classes.append(x)
for x in lines_test:
    x = x.split('\t')[0]
    test_classes.append(x)
for x in lines_all:
    x = x.split("\t")
    if x[1] in train_classes and x[0] not in lines_gray:
        train_images.append(x[0])
    elif x[1] in val_classes and x[0] not in lines_gray:
        val_images.append(x[0])
    elif x[1] in test_classes and x[0] not in lines_gray:
        test_images.append(x[0])
print(len(train_images),len(val_images),len(test_images))
z = 0
for x in test_images:
    if x in train_images or x in val_images:
        z += 1
print(z)


# ### Dataset

# In[15]:


with open(json_annotation) as f: annotations = json.load(f)
    
#with open(DATA_SPLIT) as f: DATA_SPLIT = json.load(f)
DATA_SPLIT = dict()
DATA_SPLIT["train"] = train_images
DATA_SPLIT["val"] = val_images
DATA_SPLIT["test"] = test_images


# In[16]:

train_c = CountingDataset(
    data_split = DATA_SPLIT["train"],
    split_type = "train",
    annotations = annotations,
    num_examples = 3,
    example_type = EXAMPLE_TYPE,
    do_cache = False,
    transform = transforms.Compose([
        ResizeImage(MAX_HW,crop=False),
        ToTensor(dots=False),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #CropImage(crop=False,dots=True)
    ]),src_dot_dir=src_dot_dir)


val_c = CountingDataset(
    data_split = DATA_SPLIT["val"],
    split_type = "val",
    annotations = annotations,
    num_examples = 3,
    example_type = EXAMPLE_TYPE,
    do_cache = False,
    transform = transforms.Compose([
        ResizeImage(MAX_HW,crop=False),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

train_dataloader = DataLoader(train_c, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_c, batch_size=1, shuffle=False)




best_mae, best_rmse = 1e7,1e7
stats = []
step1 = 0
start_epoch = 0

resume_param = False
if resume_param:
    #model_name = outdir + '/' + "{}_current.pth".format(MODEL_NAME_PREFIX)
    model_name = outdir + '/' + "{}_current.pth".format(MODEL_NAME_PREFIX)
    #torch.save(regressor.state_dict(), model_name)
    regressor.load_state_dict(torch.load(model_name))
    model_name1 = outdir + '/' + "Optimizer_{}_current.pth".format(MODEL_NAME_PREFIX)
    optimizer.load_state_dict(torch.load(model_name))
    model_name = outdir + '/' + "{}_current_State.pth".format(MODEL_NAME_PREFIX)
    Ad = torch.load(model_name)
    start_epoch = Ad['epoch'] + 1
    best_mae = Ad['best_mae']
    best_rmse = Ad['best_rmse']
    optimizer.load_state_dict(torch.load(model_name1))

print("Check")
for epoch in range(start_epoch,MAX_EPOCHS):
    
    t0 = time.time()
    
    """
    Training
    """   
    step = epoch
    regressor.train()
    resnet50_conv.eval()
    train_epoch_loss = 0
    train_mae = 0
    train_rmse = 0
    for _, sample_batched in enumerate(train_dataloader):
        optimizer.zero_grad()
        '''
        box_sum = sample_batched['boxes'].sum(dim=-1).sum(dim=-1)
        non_zero_id = torch.nonzero(box_sum.squeeze()).squeeze().tolist()
        if isinstance(non_zero_id, int):
            non_zero_id = [non_zero_id]
        if len(non_zero_id) < BATCH_SIZE:
            for j in range(0,sample_batched['boxes'].shape[0]): # replace bounding box outside the crop with one from inside
                if sample_batched['boxes'][j].sum() == 0:
                    for key in sample_batched.keys():
                        sample_batched[key][j] = sample_batched[key][non_zero_id[0]]         
        '''
        image, boxes = sample_batched['image'].cuda(), sample_batched['boxes'].cuda()
        pathX = sample_batched['path'][0]

        boxes = All_Proposal_new[pathX][0:NUM_EXAMPLES].cuda().unsqueeze(0).unsqueeze(0)
        All_boxes = All_Proposal_Boxes[pathX].cuda()
        All_ious = box_iou(boxes.squeeze()[:,1:5],All_boxes.squeeze()[:,1:5])
        loss_real_count = 0.
        pos_list = list()
        for ix in range(0,All_ious.shape[0]):
            max_iou = All_ious[ix].max().item()
            if max_iou > iou_thresh:   
                pos_list.append(ix)
        #print(pos_list,boxes.shape)
        if len(pos_list) > 0 and len(pos_list) < NUM_EXAMPLES:
            for ix in range(0,All_ious.shape[0]):
                if ix not in pos_list:
                    if len(pos_list) > 1:
                        choice = randint(0,len(pos_list)-1)
                    else:
                        choice = 0
                    boxes[0,0,ix,:] = boxes[0,0,pos_list[choice],:]

        #noise = torch.randn(image.shape[0],128).cuda()
        density = sample_batched['density'].cuda()
        ################   real loss   ###############################
        with torch.no_grad(): features0 = extract_features(resnet50_conv,image,boxes,MAPS,Scales)
        if len(pos_list) == 0:
            teacher_output = teacher_FamNet(features0).detach()            
        features0.requires_grad = True
        rpn_count = All_Counts_new[pathX][0:NUM_EXAMPLES].cuda()
        rpn_map = torch.ones(1,NUM_EXAMPLES,1,features0.shape[-2],features0.shape[-1]).cuda()
        for i1 in range(0,NUM_EXAMPLES):
            rpn_map[0,i1] = rpn_map[0,i1] * rpn_count[i1]
        #features0 = torch.rand(1,3,6,2,2)
        #count_map = torch.rand(1,3,1,2,2)
        features0 = torch.cat((features0,rpn_map),dim=2)        
        #density_mixup = density[index,:]
        #density = lam * density + (1 - lam) * density_mixup
        #print(features0.shape)
        output0 = regressor(features0)
        if len(pos_list) == 0:
            #teacher_output = teacher_FamNet(features0).detach()
            loss_real_count =  ((wRealCount/1.) * loss(output0, teacher_output))
        else:
            loss_real_count = wRealCount * loss(output0, density)
        ####### backprop counting network
        total_loss =  (loss_real_count )/(1.0 * BATCH_SIZE)
        total_loss.backward()
        optimizer.step()
        ####### compute discriminator loss
        #optimizer_D.zero_grad()
        ##############################  
        train_epoch_loss += loss_real_count.item()
        ##############################
        cnt_err = abs(torch.sum(output0).item() - torch.sum(density).item())
        train_mae += cnt_err
        train_rmse += cnt_err ** 2      
        step1 += 1          
    """
    Validation Error
    """  
    regressor.eval()
    #generator.eval()
    resnet50_conv.eval()
    #netFeat.eval()
    val_mae = 0
    val_rmse = 0
    for _, sample_batched in enumerate(val_dataloader):
        with torch.no_grad():
            imageA = sample_batched['image'].cuda()
            density0 = sample_batched['density'].cuda()
            pathX = sample_batched['path'][0]
            boxes = All_Proposal_new[pathX][0:3].cuda().unsqueeze(0).unsqueeze(0)
            #boxes = boxes.unsqueeze(0)
            #print(boxes.shape,'Test')
            with torch.no_grad(): features0 = extract_features(resnet50_conv,imageA,boxes,MAPS,Scales) 
            rpn_count = All_Counts_new[pathX][0:NUM_EXAMPLES].cuda()
            rpn_map = torch.ones(1,NUM_EXAMPLES,1,features0.shape[-2],features0.shape[-1]).cuda()
            for i1 in range(0,NUM_EXAMPLES):
                rpn_map[0,i1] = rpn_map[0,i1] * rpn_count[i1]
            #features0 = torch.rand(1,3,6,2,2)
            #count_map = torch.rand(1,3,1,2,2)
            features0 = torch.cat((features0,rpn_map),dim=2)                
            with torch.no_grad():output0 = regressor(features0)#regressor(features0)
            ################# Generator loss    ##################################
            cnt_err = abs(output0.sum().item() - density0.sum().item())
            val_mae += cnt_err
            val_rmse += cnt_err ** 2
    
    """
    Stats Computation
    """
    val_mae = (val_mae / val_c.__len__())
    val_rmse = (val_rmse / val_c.__len__())**0.5
    train_epoch_loss = train_epoch_loss / len(train_dataloader)
    train_mae = (train_mae / train_c.__len__())
    train_rmse = (train_rmse / train_c.__len__())**0.5
    #train_consistency_loss = train_consistency_loss / len(train_dataloader)
    #train_center_loss = train_center_loss / len(train_dataloader)  
    stats.append((train_epoch_loss, train_mae, train_rmse, val_mae, val_rmse))
    stats_file = join(outdir, "stats" +  ".txt")
    #print(stats_file)
    with open(stats_file, 'w') as f:
        for s in stats:
            f.write("%s\n" % ','.join([str(x) for x in s]))    
    f.close()
    train_epoch_loss = 0
    """
    Saving Best Model
    """
    if best_mae >= val_mae:
        best_mae = val_mae
        best_rmse = val_rmse
        model_name = outdir + '/' + "{}_best.pth".format(MODEL_NAME_PREFIX)
        torch.save(regressor.state_dict(), model_name)      
    t1 = time.time()
    model_name = outdir + '/' + "{}_current.pth".format(MODEL_NAME_PREFIX)
    torch.save(regressor.state_dict(), model_name)
    model_name = outdir + '/' + "Optimizer_{}_current.pth".format(MODEL_NAME_PREFIX)
    torch.save(optimizer.state_dict(), model_name)
    current_state = {'epoch':epoch,'best_mae':best_mae,'best_rmse':best_rmse}
    model_name = outdir + '/' + "{}_current_State.pth".format(MODEL_NAME_PREFIX)
    torch.save(current_state, model_name)

    print("ZSC  3 proposals DenseRPN KD FamNet B Binarized KT Mean Pooled C Sorted Count RPN Map Save 2: {} Model: {} Avg. Epoch Loss: {} Train MAE: {} Train RMSE: {} Val MAE: {} Val RMSE: {} Best Val MAE: {} Best Val RMSE: {} Time taken: {} secs".format(
              epoch+1, MODEL_NAME_PREFIX, stats[-1][0], stats[-1][1], stats[-1][2], stats[-1][3], stats[-1][4], best_mae, best_rmse, round((t1-t0))))



