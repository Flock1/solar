#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from fastai import *
import csv
from glob import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import fastai
from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.schedule import lr_find
from torch import nn
import torch.nn.functional as F


# In[5]:


class UpSample(nn.Module):
    def __init__(self,feat_in,feat_out,out_shape=None,scale=2):
        super().__init__()
        self.conv = nn.Conv2d(feat_in,feat_out,kernel_size=(3,3),stride=1,padding=1)
        self.out_shape,self.scale = out_shape,scale
        
    
    def forward(self,x):
        return self.conv(
            nn.functional.interpolate(
                x,size=self.out_shape,scale_factor=self.scale,mode='bilinear',align_corners=True))

def get_upSamp(feat_in,feat_out, out_shape=None, scale=2, act='relu'):
    
    upSamp = UpSample(feat_in,feat_out,out_shape=out_shape,scale=scale).cuda()
    
    layer = nn.Sequential(upSamp)
    
    if act == 'relu':
        act_f = nn.ReLU(inplace=True).cuda()
        bn = nn.BatchNorm2d(feat_out).cuda()
        layer.add_module('ReLU',act_f)
        layer.add_module('BN',bn)
    elif act == 'sig':
        act_f = nn.Sigmoid()
        layer.add_module('Sigmoid',act_f)
    return layer

def add_layer(m,feat_in,feat_out,name,out_shape=None,scale=2,act='relu'):
    upSamp = get_upSamp(feat_in,feat_out,out_shape=out_shape,scale=scale,act=act)
    m.add_module(name,upSamp)


# In[7]:


def test_model():
    m = fastai.vision.models.resnet34(pretrained = False).cuda()
    m = nn.Sequential(*list(m.children())[:-3])
    code_sz = 32

    conv = nn.Conv2d(256, code_sz, kernel_size=(2,2)).cuda()

    m.add_module('CodeIn',conv)
    m._modules['0'] = nn.Conv2d(1, 64, kernel_size=(7,7),stride=2,padding=1).cuda()
    add_layer(m,code_sz,256,'CodeOut',out_shape=(64,64),scale=None)
    add_layer(m,256,128,'Upsample0')
    add_layer(m,128,64,'Upsample1')
    add_layer(m,64,32,'Upsample2')
    add_layer(m,32,1,'Upsample3',act='sig', scale=1)
    
    return m

