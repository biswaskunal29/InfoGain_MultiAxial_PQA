import csv
from pathlib import Path
import numpy as np
#import Final_token_reader as tr
#from sentence_transformers import SentenceTransformer


from Final_image_vit_encoder import get_image_vit_enc
from Final_image_Text_encoder import get_image_text_enc
from Final_dp_vit_encoder import get_dp_vit_enc
from Final_banner_vit_encoder import get_banner_vit_enc
from Final_profile_Text_encoder import get_profile_text_enc


#model = SentenceTransformer('all-MiniLM-L6-v2')

def save_image_vit_enc(uid,imgid,new_dir):
    
    enc = get_image_vit_enc(uid,imgid, wavelet = 1, rgb = 0, way1 = 0, way2 = 1)
    print(enc.shape)
#    filename = uid + '_' + imgid + part1 + '.npy'
#    filepath = new_dir / filename
#    np.save(filepath, enc)
    
def save_image_text_enc(uid,imgid,new_dir):
    
    enc = get_image_text_enc(uid,imgid, infobert = 0)
    print(enc.shape)
#    filename = uid + '_' + imgid + part2 + '.npy'
#    filepath = new_dir / filename
#    np.save(filepath, enc)   
    
def save_dp_vit_enc(uid,imgid,new_dir):
    
    enc = get_dp_vit_enc(uid,imgid, wavelet = 1, rgb = 0, way1 = 0, way2 = 1)
    print(enc.shape)
#    filename = uid + '_' + imgid + part3 + '.npy'
#    filepath = new_dir / filename
#    np.save(filepath, enc)  
    
def save_banner_vit_enc(uid,imgid,new_dir):
    
    enc = get_banner_vit_enc(uid,imgid, wavelet = 1, rgb = 0, way1 = 0, way2 = 1)
    print(enc.shape)
#    filename = uid + '_' + imgid + part4 + '.npy'
#    filepath = new_dir / filename
#    np.save(filepath, enc)   
    
def save_profile_text_enc(uid,imgid,new_dir):
    
    enc = get_profile_text_enc(uid,imgid, infobert = 0)
    print(enc.shape)
#    filename = uid + '_' + imgid + part5 + '.npy'
#    filepath = new_dir / filename
#    np.save(filepath, enc)     
    

# =============================================================================
#     image_ocr = tr.get_image_HW_tokens(uid,imgid)
# #    print(len(image_ocr))
#     image_ocr_emb = model.encode(image_ocr)
# #    print(image_ocr_emb.shape)
#     filename = sample_label + part1 + '.npy'
#     filepath = new_dir / filename
#     np.save(filepath, image_ocr_emb)
# =============================================================================


df_csv_file = "Dataset_uniqid_class.csv"
FOLDER = r'E:\CVPR 23\Datasets\Our Dataset 5 Class\Ablation 6 Wavelet and way2 only'

part1 = '_image_vit'
part2 = '_image_text'
part3 = '_dp_vit'
part4 = '_banner_vit'
part5 = '_profile_text'

#part2 = '_dp_ocr'
#part5 = '_dp_labels'
#part6 = '_banner_labels'
#part7 = '_image_desc'
#part8 = '_profile_desc'


with open(df_csv_file, mode = 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

for i in range(2):
#    print(data[i][0],data[i][1])
#    print(i)
    
    sample_label = data[i][0]
    class_label = data[i][1] 
#    print(sample_label)
#    print(class_label)
    
    
    parts = [x.strip() for x in sample_label.split("_")]
    (uid,imgid) = parts
#    print(uid)
#    print(imgid)
    
    directory = sample_label
    new_dir = Path(FOLDER,directory)
    new_dir.mkdir(parents=True, exist_ok=True)
    
    save_image_vit_enc(uid,imgid,new_dir)
    save_image_text_enc(uid,imgid,new_dir)
    save_dp_vit_enc(uid,imgid,new_dir)
    save_banner_vit_enc(uid,imgid,new_dir)
    save_profile_text_enc(uid,imgid,new_dir)
    
#    print(str(i) + "//" + str(5001))   
    
    
   

