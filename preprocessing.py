#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to preprocess the input image data. The image is converted to 
grayscale and normalized

@author: Varun Venkatesh
"""
import numpy as np

def preprocessing(image):
    
    #Grayscaling the image
    gray = np.sum(image/3, axis=3, keepdims=True)
    
    #Normalizing the gray images
    normalize = (gray - 128)/128
    
    return normalize
