import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.loss.ntx_ent_loss import NTXentLoss
from get_contrastive_images import sim_pair,dissim_pair
import cv2



def preprocess(img):
    img_flat = img.flatten()
    img_flat_tor = torch.from_numpy(img_flat)
    img_flat_tor_float = img_flat_tor.to(torch.float)
    mean, std = torch.mean(img_flat_tor_float), torch.std(img_flat_tor_float)
    img_flat_tor_float_norm  = (img_flat_tor_float-mean)/std
#    img_flat_tor_float_norm_device = img_flat_tor_float_norm.to(device)
    
    
    return img_flat_tor_float_norm

def get_image_enc(uid,imgid, image_features = 1):         # image_features = 0 if no need of image features
    
    readfile = FOLDER + '\\' + uid + '\\' + imgid + '.jpg'
    
    
    
    
#    try:
    rgb_image = cv2.imread(readfile)
    if rgb_image is None:
#    if(rgb_image.any() == None):
        print("Got no image here", uid, imgid)
        enc = torch.zeros(64, dtype = torch.float32)
        return enc
    
    rgb_image = cv2.resize(rgb_image, (64, 64))
    grey_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
#    grey_image = cv2.imread(readfile,0)
#    grey_image = cv2.resize(grey_image, (image_len,image_len))
#    except:
#        enc = torch.zeros(4192, dtype = torch.float32)
#        return enc
    
    cv2.imshow("grey image", grey_image)
    
    
    
    
    
    
    
    
#df_csv_file = "Dataset_uniqid_class.csv"
FOLDER = r'F:\PhD\Datasets\twitter-collection-master\twitter-collection-master\Final Dataset v11'

readfile = "96573740_36 neur.jpg"

if __name__ == "__main__":
    
    uid = "749003"
    imgid = '3'

    enc = get_image_enc(uid,imgid)
#    enc = get_image_vit_enc(uid,imgid, wavelet = 1, rgb = 1, way1 = 1, way2 = 1)
#    enc = get_image_vit_enc(uid,imgid)
    
    print(enc.shape)
    print(np.max(enc))
    print(np.min(enc))
#    print(enc[2120])
#    print(enc[100])











