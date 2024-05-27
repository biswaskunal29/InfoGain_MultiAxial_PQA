import Final_token_reader as tr
import json
#import re
import torch
import numpy as np
from sys import getsizeof
import csv

def read_class():
    with open(class_file, newline='\n', encoding='utf-8') as csv_file:
        cf = csv.reader(csv_file, delimiter=',',quotechar='"')
        
#        dataset = {}
        for row in cf:
            if(row[0]=='user_id'): continue
            class_dict[row[0]] = row[1]
 
def getB1(uniqid):
    beta1 = float(B1[class_dict[uniqid]])
    beta2 = float(B2[class_dict[uniqid]])
    beta3 = float(B3[class_dict[uniqid]])
    sum_beta = float(beta1 + beta2 + beta3)
    return beta1/sum_beta

def getB2(uniqid):
    beta1 = float(B1[class_dict[uniqid]])
    beta2 = float(B2[class_dict[uniqid]])
    beta3 = float(B3[class_dict[uniqid]])
    sum_beta = float(beta1 + beta2 + beta3)
    return beta2/sum_beta

def getB3(uniqid):
    beta1 = float(B1[class_dict[uniqid]])
    beta2 = float(B2[class_dict[uniqid]])
    beta3 = float(B3[class_dict[uniqid]])
    sum_beta = float(beta1 + beta2 + beta3)
    return beta3/sum_beta

def getB4(uniqid):
    beta4 = float(B4[class_dict[uniqid]])
    beta5 = float(B5[class_dict[uniqid]])
    beta6 = float(B6[class_dict[uniqid]])
    sum_beta = float(beta4 + beta5 + beta6)
    return beta4/sum_beta

def getB5(uniqid):
    beta4 = float(B4[class_dict[uniqid]])
    beta5 = float(B5[class_dict[uniqid]])
    beta6 = float(B6[class_dict[uniqid]])
    sum_beta = float(beta4 + beta5 + beta6)
    return beta5/sum_beta

def getB6(uniqid):
    beta4 = float(B4[class_dict[uniqid]])
    beta5 = float(B5[class_dict[uniqid]])
    beta6 = float(B6[class_dict[uniqid]])
    sum_beta = float(beta4 + beta5 + beta6)
    return beta6/sum_beta

def getB7(uniqid):
    beta7 = float(B7[class_dict[uniqid]])
    beta8 = float(B8[class_dict[uniqid]])
    sum_beta = float(beta7 + beta8)
    return beta7/sum_beta

def getB8(uniqid):
    beta7 = float(B7[class_dict[uniqid]])
    beta8 = float(B8[class_dict[uniqid]])
    sum_beta = float(beta7 + beta8)
    return beta8/sum_beta

def get_image_HW_arr(uid,imgid):
    token_list = tr.get_image_HW_tokens(uid,imgid)
#    print(type(token_list),token_list)
    
    word_index_list = [vocab_dict.get(item,-1)  for item in token_list]
#    print(word_index_list)
    
    #word_index_list = word_index_list[:img_HW_size]
    word_index_list_int = list(map(int, word_index_list))
#    print(f"{len(word_index_list_int)}\n{word_index_list_int}")
    
    word_index_list_int = word_index_list_int[:img_HW_size]
    if len(word_index_list_int) < img_HW_size:
        word_index_list_int += [pad_val] * (img_HW_size - len(word_index_list_int))   
#    print(f"{len(word_index_list_int)}\n{word_index_list_int}")
        
    curr_arr = np.ones(25)
    #print(curr_arr)      
    for item in word_index_list_int:
        if item == -1:
            curr_arr = np.vstack((curr_arr,np.zeros(25)))
        
        else:
            curr_arr = np.vstack((curr_arr,word_vec_arr[item]))
    #    break
    curr_arr = np.delete(curr_arr,0,0)        
    
    #print(curr_arr) 
    curr_arr = curr_arr.flatten()     
#    print(curr_arr)
#    print(curr_arr.shape)         
    #print(getsizeof(curr_arr)) 
    
    curr_arr = curr_arr * img_HW_multi
#    print(curr_arr)
#    print(curr_arr.shape)         
#    print(getsizeof(curr_arr)) 
    return curr_arr
    
def get_image_label_arr(uid,imgid):
    token_list = tr.get_image_labels_tokens(uid,imgid)  
    word_index_list = [vocab_dict.get(item,-1)  for item in token_list]
    word_index_list_int = list(map(int, word_index_list))   
    word_index_list_int = word_index_list_int[:other_size]
    if len(word_index_list_int) < other_size:
        word_index_list_int += [pad_val] * (other_size - len(word_index_list_int))       
    curr_arr = np.ones(25)     
    for item in word_index_list_int:
        if item == -1:
            curr_arr = np.vstack((curr_arr,np.zeros(25)))        
        else:
            curr_arr = np.vstack((curr_arr,word_vec_arr[item]))
    curr_arr = np.delete(curr_arr,0,0)        
    curr_arr = curr_arr.flatten()        
    curr_arr = curr_arr * img_label_multi 
    return curr_arr   

def get_image_desc_arr(uid,imgid):
    token_list = tr.get_image_desc_tokens(uid,imgid)  
    word_index_list = [vocab_dict.get(item,-1)  for item in token_list]
    word_index_list_int = list(map(int, word_index_list))   
    word_index_list_int = word_index_list_int[:other_size]
    if len(word_index_list_int) < other_size:
        word_index_list_int += [pad_val] * (other_size - len(word_index_list_int))       
    curr_arr = np.ones(25)     
    for item in word_index_list_int:
        if item == -1:
            curr_arr = np.vstack((curr_arr,np.zeros(25)))        
        else:
            curr_arr = np.vstack((curr_arr,word_vec_arr[item]))
    curr_arr = np.delete(curr_arr,0,0)        
    curr_arr = curr_arr.flatten()        
    curr_arr = curr_arr * img_desc_multi 
    return curr_arr

def get_profile_HW_arr(uid,imgid):
    token_list = tr.get_profile_HW_tokens(uid,imgid)  
    word_index_list = [vocab_dict.get(item,-1)  for item in token_list]
    word_index_list_int = list(map(int, word_index_list))   
    word_index_list_int = word_index_list_int[:other_size]
    if len(word_index_list_int) < other_size:
        word_index_list_int += [pad_val] * (other_size - len(word_index_list_int))       
    curr_arr = np.ones(25)     
    for item in word_index_list_int:
        if item == -1:
            curr_arr = np.vstack((curr_arr,np.zeros(25)))        
        else:
            curr_arr = np.vstack((curr_arr,word_vec_arr[item]))
    curr_arr = np.delete(curr_arr,0,0)        
    curr_arr = curr_arr.flatten()        
    curr_arr = curr_arr * prf_HW_multi 
    return curr_arr    

def get_profile_label_arr(uid,imgid):
    token_list = tr.get_profile_labels_tokens(uid,imgid) 
    word_index_list = [vocab_dict.get(item,-1)  for item in token_list]
    word_index_list_int = list(map(int, word_index_list))   
    word_index_list_int = word_index_list_int[:other_size]
    if len(word_index_list_int) < other_size:
        word_index_list_int += [pad_val] * (other_size - len(word_index_list_int))       
    curr_arr = np.ones(25)     
    for item in word_index_list_int:
        if item == -1:
            curr_arr = np.vstack((curr_arr,np.zeros(25)))        
        else:
            curr_arr = np.vstack((curr_arr,word_vec_arr[item]))
    curr_arr = np.delete(curr_arr,0,0)        
    curr_arr = curr_arr.flatten()        
    curr_arr = curr_arr * prf_label_multi 
    return curr_arr

def get_profile_desc_arr(uid,imgid):
    token_list = tr.get_profile_desc_tokens(uid,imgid) 
    word_index_list = [vocab_dict.get(item,-1)  for item in token_list]
    word_index_list_int = list(map(int, word_index_list))   
    word_index_list_int = word_index_list_int[:other_size]
    if len(word_index_list_int) < other_size:
        word_index_list_int += [pad_val] * (other_size - len(word_index_list_int))       
    curr_arr = np.ones(25)     
    for item in word_index_list_int:
        if item == -1:
            curr_arr = np.vstack((curr_arr,np.zeros(25)))        
        else:
            curr_arr = np.vstack((curr_arr,word_vec_arr[item]))
    curr_arr = np.delete(curr_arr,0,0)        
    curr_arr = curr_arr.flatten()        
    curr_arr = curr_arr * prf_desc_multi 
    return curr_arr

def get_banner_HW_arr(uid,imgid):
    token_list = tr.get_banner_HW_tokens(uid,imgid) 
    word_index_list = [vocab_dict.get(item,-1)  for item in token_list]
    word_index_list_int = list(map(int, word_index_list))   
    word_index_list_int = word_index_list_int[:other_size]
    if len(word_index_list_int) < other_size:
        word_index_list_int += [pad_val] * (other_size - len(word_index_list_int))       
    curr_arr = np.ones(25)     
    for item in word_index_list_int:
        if item == -1:
            curr_arr = np.vstack((curr_arr,np.zeros(25)))        
        else:
            curr_arr = np.vstack((curr_arr,word_vec_arr[item]))
    curr_arr = np.delete(curr_arr,0,0)        
    curr_arr = curr_arr.flatten()        
    curr_arr = curr_arr * ban_HW_multi 
    return curr_arr

def get_banner_label_arr(uid,imgid):
    token_list = tr.get_banner_labels_tokens(uid,imgid) 
    word_index_list = [vocab_dict.get(item,-1)  for item in token_list]
    word_index_list_int = list(map(int, word_index_list))   
    word_index_list_int = word_index_list_int[:other_size]
    if len(word_index_list_int) < other_size:
        word_index_list_int += [pad_val] * (other_size - len(word_index_list_int))       
    curr_arr = np.ones(25)     
    for item in word_index_list_int:
        if item == -1:
            curr_arr = np.vstack((curr_arr,np.zeros(25)))        
        else:
            curr_arr = np.vstack((curr_arr,word_vec_arr[item]))
    curr_arr = np.delete(curr_arr,0,0)        
    curr_arr = curr_arr.flatten()        
    curr_arr = curr_arr * ban_label_multi 
    return curr_arr

def get_full_arr(uid,imgid):
    
    img_HW_arr = get_image_HW_arr(uid,imgid)
    img_label_arr = get_image_label_arr(uid,imgid)
    img_desc_arr = get_image_desc_arr(uid,imgid)
    
    profile_HW_arr = get_profile_HW_arr(uid,imgid) *0.0
    profile_label_arr = get_profile_label_arr(uid,imgid) *0.0
    profile_desc_arr = get_profile_desc_arr(uid,imgid) *0.0
    
    banner_HW_arr = get_banner_HW_arr(uid,imgid) *0.0
    banner_label_arr = get_banner_label_arr(uid,imgid) *0.0
    
    full_array = np.hstack((img_HW_arr, img_label_arr, img_desc_arr, profile_HW_arr, profile_label_arr, profile_desc_arr, banner_HW_arr, banner_label_arr))
    
    #print(profile_HW_arr[:10])
    #print(profile_HW_arr.shape)
    
#    print(full_array[11750:11760])
#    print(full_array.shape)
#    print(getsizeof(full_array))
    return full_array

def get_class(uniqid):
    class_label = class_dict[uniqid]
    class_2_num = {'O':0, 'C':1, 'E':2, 'A':3, 'N':4}
    class_vec = np.zeros(5)
    class_vec[class_2_num[class_label]] = 1
    return class_vec

# =============================================================================
# All functions Done
# =============================================================================

json_file = "new_vocab_to_index_dict.json"
weight_file = "new_weight_file.pt"
class_file = "Dataset_uniqid_class.csv"
class_dict = {}

img_HW_size = 400
other_size = 35
pad_val = -1

# =============================================================================
#            loading vocab Dict
# =============================================================================
with open(json_file) as json_file_temp: 
    d = json.load(json_file_temp)
#print(len(d))
vocab_dict = { v:k for k,v in d.items()}
#print(len(vocab_dict))

# =============================================================================
#            Loading GloVe vectors
# =============================================================================
word_vec_gpu = torch.load(weight_file)
word_vec_gpu.requires_grad_(False)
word_vec = word_vec_gpu.to(device = 'cpu')
word_vec_arr = word_vec.numpy()
#print(word_vec_arr)
#print(type(word_vec_arr))
#print(word_vec_arr.shape)
#print(word_vec[10])

# =============================================================================
#            Loading Class Dict
# =============================================================================
read_class()
#print(class_dict[uniqid])

# =============================================================================
#            Setting Classwise Betas
# =============================================================================
B1 = {'O': 20, 'C': 20, 'E': 19, 'A': 18, 'N': 21}
B2 = {'O': 3, 'C': 1, 'E': 5, 'A': 3, 'N': 4}
B3 = {'O': 6, 'C': 7, 'E': 7, 'A': 6, 'N': 5}
B4 = {'O': 1, 'C': 1, 'E': 1, 'A': 1, 'N': 1}
B5 = {'O': 8, 'C': 5, 'E': 7, 'A': 7, 'N': 8}
B6 = {'O': 11, 'C': 10, 'E': 7, 'A': 11, 'N': 8}
B7 = {'O': 1, 'C': 1, 'E': 1, 'A': 1, 'N': 1}
B8 = {'O': 3, 'C': 3, 'E': 3, 'A': 4, 'N': 2}

#print(B1['N'])
#print(B1[class_dict[uniqid]])
#print(B2[class_dict[uniqid]])


'''
length of each part will be

B1      image_HW        = 860   400
B2      image_labels    = 22    35
B3      image_desc      = 27    35
B4      profile_HW      = 14    35
B5      profile_labels  = 13    35
B6      profile_desc    = 33    35
B7      banner_HW       = 25    35
B8      banner_labels   = 16    35

sum                     =       645

'''

# uniqid = "749003_7"
# uid,imgid = uniqid.split('_',1)

list_file = "dataset_image_list v14.txt"
fail_list_file = "read_X_y_fail.txt"
#try_output_file = "try_banner_dataset_beta_class.csv"
#output_file = "banner_dataset_beta_class.csv"
new_status = "X_y_loaded_v3"

X_curr = np.zeros(16125)
y_curr = np.zeros(5)

for target_line in range(5003): 
    
    with open(list_file, "r") as imagelist_file:
        linelist = imagelist_file.readlines()
    
    parts = [x.strip() for x in linelist[target_line].split(",")]
    (uid,imgid,status) = parts
    
    if linelist[target_line].startswith("#"): continue
    if(status == new_status): 
    #        print("Done")
        continue
    
#    try: 
    #do whatever
    uniqid = uid + "_" + imgid
    
    img_HW_multi = getB1(uniqid)
    img_label_multi = getB2(uniqid)
    img_desc_multi = getB3(uniqid)
    prf_HW_multi = getB4(uniqid)
    prf_label_multi = getB5(uniqid)
    prf_desc_multi = getB6(uniqid)
    ban_HW_multi = getB7(uniqid)
    ban_label_multi = getB8(uniqid)    
    #print(img_HW_multi,img_label_multi,img_desc_multi)
    
    X_curr = np.vstack((X_curr,get_full_arr(uid,imgid)))
    #X_curr = get_full_arr(uid,imgid)
    
#    print(X_curr)
#    print(X_curr.shape)
    #print(getsizeof(X_curr))
    
    y_curr = np.vstack((y_curr,get_class(uniqid)))
    
#    print(y_curr)
#    print(y_curr.shape)
    
# =============================================================================
#     with open(output_file, mode='a', newline='\n') as op_file:
#         op_writer = csv.writer(op_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
#         op_writer.writerow([uniqid, beta_dict[uniqid]['b7'], beta_dict[uniqid]['b8'], class_dict[uniqid]]) 
# =============================================================================
        
    print(f"X and y added of user id : {uid}\t imgid : {imgid}\n")
    
    
    
# =============================================================================
#     except:    
#         with open(fail_list_file, mode = 'a') as failer:
#             fail_text = f"{uid},{imgid}.jpg\n"
#             failer.write(fail_text)
# =============================================================================
    
    #all updations to files
    linelist[target_line] = str(uid) + "," + str(imgid) + "," + new_status +"\n"
    with open(list_file, "w") as imagelist_file:
        imagelist_file.writelines(linelist)


X_curr = np.delete(X_curr,0,0)
y_curr = np.delete(y_curr,0,0)

#np.save('full_X_v3_data.npy', X_curr)
#np.save('full_y_v3_data.npy', y_curr)

print(f"Final shape of X_curr: {X_curr.shape} and taking {getsizeof(X_curr)} bytes")
print(f"Final shape of y_curr: {y_curr.shape} and taking {getsizeof(y_curr)} bytes")

#print(f"Final X_curr is {X_curr} \nof shape : {X_curr.shape} and taking {getsizeof(X_curr)} bytes")
#print(f"Final y_curr is {y_curr} \nof shape : {y_curr.shape} and taking {getsizeof(y_curr)} bytes")










# =============================================================================
# uniqid = "749003_7"
# uid,imgid = uniqid.split('_',1)
# 
# img_HW_multi = getB1(uniqid)
# img_label_multi = getB2(uniqid)
# img_desc_multi = getB3(uniqid)
# #print(img_HW_multi,img_label_multi,img_desc_multi)
# 
# X_curr = np.vstack((X_curr,get_full_arr(uid,imgid)))
# #X_curr = get_full_arr(uid,imgid)
# 
# print(X_curr)
# print(X_curr.shape)
# #print(getsizeof(X_curr))
# 
# y_curr = np.vstack((y_curr,get_class(uniqid)))
# 
# print(y_curr)
# print(y_curr.shape)
# 
# 
# 
# X_curr = np.delete(X_curr,0,0)
# y_curr = np.delete(y_curr,0,0)
# =============================================================================