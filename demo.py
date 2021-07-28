#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 11:15:22 2021

@author: ahmedbilal
"""
import matplotlib.pyplot as plt
from inference import preTrainedResUnet_inference
from PIL import Image
import torchvision.transforms as transforms
import torch

demo_im = Image.open('/Users/ahmedbilal/Desktop/DL_Final_Project/demo.png')
demo_im_out = '/Users/ahmedbilal/Desktop/DL_Final_Project/demo_out.png'
to_tensor = transforms.ToTensor()
to_image = transforms.ToPILImage()
demo_im = to_tensor(demo_im)
#strip transparency layer
demo_im = demo_im[0:3, :, :]


inference = preTrainedResUnet_inference(output_channels=2)

out = inference.run_inference(demo_im)
if inference.output_channels == 2:
    out = (torch.lt(out[0, :, :],out[1, :, :])).float()
out = to_image(out)

out.save(fp = demo_im_out)
print('Saved!')