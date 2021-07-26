#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 11:06:57 2021

@author: ahmedbilal
"""
from models.resunet34_pretrained import load_model,resnet_pretrained
import torchvision.transforms.functional as TF



class unet_inference():
    def __init__(self, path_to_state_dict=None):
        self.model = resnet_pretrained
        self.model.eval()
    
    def run_inference(self, input):
        """Runs inference on input data
        
        input: 3x224x224 Tensor (not normalized)
        
        Returns 2x224x224 tensor with softmax probabilities where 0 channel is
        prob(~building) and 1 channel is prob(building)."""
        assert(load_model == True)
        assert(len(input.shape)==3)
        #transform
        input = TF.normalize(input, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input = input.unsqueeze(0)
        output = self.model.forward(input)
        output = output.squeeze()
        
        return output