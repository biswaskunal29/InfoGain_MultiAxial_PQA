import cv2 
import numpy as np


  
# Reading the image using imread() function
image = cv2.imread('red.jpg')
  
# Displaying the original BGR image
cv2.imshow('Original_Image', image)
  
# Using cv2.split() to split channels of coloured image 
b,g,r = cv2.split(image)
 
# =============================================================================
# r = np.zeros(image.shape, dtype = np.uint8)
# r[:,:,2] = image[:,:,2]
# 
# #r = image[:,:,2]
#  
# =============================================================================
# Displaying Blue channel image
# Blue colour is highlighted the most
cv2.imshow("Model Blue Image", b)
  
# Displaying Green channel image
# Green colour is highlighted the most
cv2.imshow("Model Green Image", g)
  
# Displaying Red channel image
# Red colour is highlighted the most
cv2.imshow("Model Red Image", r)
































