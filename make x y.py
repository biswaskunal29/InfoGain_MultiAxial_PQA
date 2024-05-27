import csv
import numpy as np
from pathlib import Path
from sys import getsizeof

def get_saved_image_vit_enc(sample_label,new_dir):
    filename = sample_label + part1 + '.npy'
    filepath = new_dir / filename
    enc = np.load(filepath)
    enc_float = enc.astype(np.single)
    return enc_float

def get_saved_image_text_enc(sample_label,new_dir):
    filename = sample_label + part2 + '.npy'
    filepath = new_dir / filename
    enc = np.load(filepath)
    enc_float = enc.astype(np.single)
    return enc_float    

def get_saved_dp_vit_enc(sample_label,new_dir):
    filename = sample_label + part3 + '.npy'
    filepath = new_dir / filename
    enc = np.load(filepath)
    enc_float = enc.astype(np.single)
    return enc_float   

def get_saved_banner_vit_enc(sample_label,new_dir):
    filename = sample_label + part4 + '.npy'
    filepath = new_dir / filename
    enc = np.load(filepath)
    enc_float = enc.astype(np.single)
    return enc_float

def get_saved_profile_text_enc(sample_label,new_dir):
    filename = sample_label + part5 + '.npy'
    filepath = new_dir / filename
    enc = np.load(filepath)
    enc_float = enc.astype(np.single)
    return enc_float

  

ydict = {'O': np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype = np.float32),
         'C': np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype = np.float32),
         'E': np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype = np.float32),
         'A': np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype = np.float32),
         'N': np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype = np.float32)}

part1 = '_image_vit'
part2 = '_image_text'
part3 = '_dp_vit'
part4 = '_banner_vit'
part5 = '_profile_text'

similarity_array = np.load("question similarities.npy")
img_ques_ans_write_csv = "Dataset_uniqid_ques_ans.csv"
FOLDER = r'.\Output'


with open(img_ques_ans_write_csv, mode = 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

X_curr = np.zeros(39072)
y_curr = np.zeros(5)

for i in range(30012):
#    print(data[i][0],data[i][1])
#    print(i)

    sample_label = data[i][0]
    quesno = int(data[i][1])
    ans = data[i][2]
#    print(sample_label)
#    print(quesno)
#    print(ans)
    
    parts = [x.strip() for x in sample_label.split("_")]
    (uid,imgid) = parts
#    print(uid, imgid, quesno, ans)
#    print(imgid)
#    print(quesno)
#    print(ans)
    
    directory = sample_label
    new_dir = Path(FOLDER,directory)
    
    image_vit_enc = get_saved_image_vit_enc(sample_label,new_dir)
    mul_1 = similarity_array[quesno][0]
    image_vit_enc_mul = mul_1 * image_vit_enc
#    print(image_vit_enc_mul.shape, mul_1)
   
    image_text_enc =  get_saved_image_text_enc(sample_label,new_dir)
    mul_2 = similarity_array[quesno][1]
    image_text_enc_mul = mul_2 * image_text_enc    
#    print(image_text_enc_mul.shape, mul_2)

#save_dp_vit_enc
       
    dp_vit_enc =  get_saved_dp_vit_enc(sample_label,new_dir)
    mul_3 = similarity_array[quesno][2]
    dp_vit_enc_mul = mul_3 * dp_vit_enc     
#    print(dp_vit_enc_mul.shape, mul_3)
       
    banner_vit_enc = get_saved_banner_vit_enc(sample_label,new_dir)
    mul_4 = similarity_array[quesno][3]
    banner_vit_enc_mul = mul_4 * banner_vit_enc
#    print(banner_vit_enc_mul.shape, mul_4)  
    
#    save_profile_text_enc(uid,imgid,new_dir)
    
    profile_text_enc = get_saved_profile_text_enc(sample_label,new_dir)
    mul_5 = similarity_array[quesno][4]
    profile_text_enc_mul = profile_text_enc
#    print(profile_text_enc_mul.shape, mul_5)  
    
    full_enc = np.hstack((image_vit_enc_mul, image_text_enc_mul, dp_vit_enc_mul, banner_vit_enc_mul, profile_text_enc_mul))
#    print(full_enc.shape)
    
    X_curr = np.vstack((X_curr,full_enc))
    y_curr = np.vstack((y_curr,ydict[ans]))
    
    print("Done till ", i , "out of  30012")
    
#    X_curr = vit_enc_mul
#    y_curr = ydict[ans]
    

X_curr = np.delete(X_curr,0,0)
y_curr = np.delete(y_curr,0,0)

#np.save('full_X_data.npy', X_curr)
#np.save('full_y_data.npy', y_curr)

print(f"Final shape of X_curr: {X_curr.shape} and taking {getsizeof(X_curr)} bytes")
print(f"Final shape of y_curr: {y_curr.shape} and taking {getsizeof(y_curr)} bytes")





















































