import csv
#from pathlib import Path
import numpy as np
import Final_token_reader as tr
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
import pickle


def get_doc(uid,imgid):
    image_ocr = tr.get_image_HW_tokens(uid,imgid)
    dp_ocr = tr.get_profile_HW_tokens(uid,imgid)
    banner_ocr = tr.get_banner_HW_tokens(uid,imgid)
    
    image_labels = tr.get_image_labels_tokens(uid,imgid)
    dp_labels = tr.get_profile_labels_tokens(uid,imgid)
    banner_labels = tr.get_banner_labels_tokens(uid,imgid)
    
    image_desc = tr.get_image_desc_tokens(uid,imgid)  
    profile_desc = tr.get_profile_desc_tokens(uid,imgid)
    
    str_list = []
    str_list.extend(image_ocr)
    str_list.extend(dp_ocr)
    str_list.extend(banner_ocr)
    
    str_list.extend(image_labels)
    str_list.extend(dp_labels)
    str_list.extend(banner_labels)
    
    str_list.extend(image_desc) 
    str_list.extend(profile_desc)   
    
    word_doc = ' '.join(str_list)
    return word_doc
    



df_csv_file = "Dataset_uniqid_class.csv"
FOLDER = r'E:\CVPR 23\Datasets\Our Dataset 5 Class\temp'
word_info_file = "word_info.pkl"

class_number = {'O':0,
                'C':1,
                'E':2,
                'A':3,
                'N':4,}

with open(df_csv_file, mode = 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

#print(type(data))


X_my = []
Y_my = np.zeros(0, dtype = np.int64)

#for i in range(5002):
for i in range(2):
    sample_label = data[i][0]
    class_label = data[i][1] 
#    print(sample_label, class_number[class_label])
#    print(class_number[class_label])\
    
    parts = [x.strip() for x in sample_label.split("_")]
    (uid,imgid) = parts
       
    
    doc = get_doc(uid,imgid)
    print(doc,'\n')
    
# =============================================================================
#     X_my.append(doc)
#     Y_my = np.hstack((Y_my,class_number[class_label]))
#     
#     print('Done till ', i)
# =============================================================================

#cv = CountVectorizer(max_df=0.95, min_df=2,
#                                     max_features=10000,
#                                     stop_words='english')
    

# =============================================================================
# #make info gain dict
# cv = CountVectorizer()
# 
# X_vec = cv.fit_transform(X_my)
# 
# res = dict(zip(cv.get_feature_names_out(),
#                mutual_info_classif(X_vec, Y_my, discrete_features=True)
#                ))
# print(len(res))
# =============================================================================


# =============================================================================
# #save the Dict
# with open(word_info_file, 'wb') as handle:
#   pickle.dump(res, handle)
# =============================================================================

# =============================================================================
# #read the dict
# with open(word_info_file, 'rb') as handle:
#   b = pickle.loads(handle.read())
# 
# print(res == b) # True
# =============================================================================






