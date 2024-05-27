import numpy as np
#import cv2
#import imutils
#import pywt
import torch
#from vit_pytorch.vit import ViT
#from vit_pytorch.extractor import Extractor
from sklearn import preprocessing
from PIL import Image
from transformers import AutoProcessor, BlipModel

model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")


def get_image_vit_enc(uid,imgid, wavelet = 1, rgb = 1, way1 = 1, way2 = 1):

            
    readfile = FOLDER + '\\' + uid + '\\' + imgid + '.jpg'  
#    readfile = filepath

    
    
    # Reading the image using imread() function
#    try:
    rgb_image = Image.open(readfile) 
    if rgb_image is None:
#    if(rgb_image.any() == None):
        print("Got no image here", uid, imgid)
        enc = torch.zeros(512, dtype = torch.float32)
        return enc
    
    
    
    inputs = processor(images=rgb_image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    image_features_np = image_features.detach().cpu().numpy()
    
    img_encoding = image_features_np.flatten()
    img_encoding = img_encoding.reshape(1, -1)
    normalized_enc = preprocessing.normalize(img_encoding)
    normalized_enc_flat = normalized_enc.flatten()
#    print(img_encoding.shape)
    
    return normalized_enc_flat


readfile = "96573740_36 neur.jpg"

FOLDER = r'F:\PhD\Datasets\twitter-collection-master\twitter-collection-master\Final Dataset v11'


if __name__ == "__main__":
#    print ("Executed when invoked directly")

    uid = "749003"
    imgid = '1'
    
    enc = get_image_vit_enc(uid,imgid)
#    enc = get_image_vit_enc(uid,imgid, wavelet = 1, rgb = 1, way1 = 1, way2 = 1)
#    enc = get_image_vit_enc(uid,imgid)
    
    print(enc.shape)
#    print(np.max(enc))
#    print(np.min(enc))
#    print(enc[2120])
#    print(enc[100])
    











