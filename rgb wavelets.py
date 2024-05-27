import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import cv2
import imutils
import pywt
import pywt.data

# =============================================================================
# def rotate(image,angle):
#     M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1) 
#     rotate = cv2.warpAffine(l,M,(cols,rows))
#     return rotate
# =============================================================================
    
def crop(image):
    y_nonzero, x_nonzero = np.nonzero(image)
#    temp = image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
#    return cv2.resize(temp, (image_len,image_len))
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

image_len = 512


# Reading the image using imread() function
image = cv2.imread("96573740_36 neur.jpg")
image = cv2.resize(image, (image_len,image_len))
l = cv2.imread("96573740_36 neur.jpg",0)
l = cv2.resize(l, (image_len,image_len)) 

  
# Using cv2.split() to split channels of coloured image 
b,g,r = cv2.split(image)
 

# Displaying the original BGR image
cv2.imshow('Original_Image', image)
#cv2.imwrite('Original_Image.jpg', image)

# =============================================================================
# # Displaying the original Luminance image
# cv2.imshow('Luminance_Image', l)
# 
# 
# # Displaying Blue channel image
# # Blue colour is highlighted the most
# cv2.imshow("Model Blue Image", b)
# cv2.imwrite('Model Blue Image.jpg', b)
#   
# # Displaying Green channel image
# # Green colour is highlighted the most
# cv2.imshow("Model Green Image", g)
# 
#   
# # Displaying Red channel image
# # Red colour is highlighted the most
# cv2.imshow("Model Red Image", r)
# =============================================================================


# =============================================================================
# #Saving Images
# cv2.imwrite('Original_Image.jpg', image)
# cv2.imwrite('Luminance_Image.jpg', l)
# cv2.imwrite('Model Blue Image.jpg', b)
# cv2.imwrite('Model Green Image.jpg', g)
# cv2.imwrite('Model Red Image.jpg', r)
# =============================================================================


#coeffs2 = pywt.dwt2(original, 'bior1.3')

#cols,rows = l.shape

#angle = 30

# =============================================================================
# M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1) 
# rotate = cv2.warpAffine(l,M,(cols,rows))
# =============================================================================

rotated = imutils.rotate_bound(l, -22.5)
#cv2.imshow("Rotated Image", rotated)

rerotated = imutils.rotate_bound(rotated, 22.5)
#cv2.imshow("re Rotated Image", rerotated)

crop_rerotated = crop(rerotated)
#cv2.imshow("crop re Rotated Image", crop_rerotated)






coeffs2 = pywt.dwt2(rotated, 'bior1.3')
coeffs1 = pywt.dwt2(l, 'bior1.3')

LL_r, (LH_r, HL_r, HH_r) = coeffs2
cv2.imshow("rotated", LH_r)

LL1, (LH1, HL1, HH1) = coeffs1
cv2.imshow("Original", LL1)

rerotated = imutils.rotate_bound(LH_r, 22.5)
cv2.imshow("re Rotated wavelet Image", rerotated)

crop_rerotated = crop(rerotated)
cv2.imshow("crop re Rotated wavelet Image", crop_rerotated)



























