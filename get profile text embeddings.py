import pickle
import Final_token_reader as tr
import csv
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer


def get_profile_words_list(uid,imgid):
#    image_ocr = tr.get_image_HW_tokens(uid,imgid)
    dp_ocr = tr.get_profile_HW_tokens(uid,imgid)
    banner_ocr = tr.get_banner_HW_tokens(uid,imgid)
    
#    image_labels = tr.get_image_labels_tokens(uid,imgid)
    dp_labels = tr.get_profile_labels_tokens(uid,imgid)
    banner_labels = tr.get_banner_labels_tokens(uid,imgid)
    
#    image_desc = tr.get_image_desc_tokens(uid,imgid)  
    profile_desc = tr.get_profile_desc_tokens(uid,imgid)
    
    str_list = []
#    str_list.extend(image_ocr)
    str_list.extend(dp_ocr)
    str_list.extend(banner_ocr)
    
#    str_list.extend(image_labels)
    str_list.extend(dp_labels)
    str_list.extend(banner_labels)
    
#    str_list.extend(image_desc) 
    str_list.extend(profile_desc)   
    
#    word_doc = ' '.join(str_list)
    return str_list



word_info_file = "word_info.pkl"
df_csv_file = "Dataset_uniqid_class.csv"
FOLDER = r'E:\CVPR 23\Datasets\Our Dataset 5 Class\temp'
profile_average_words = 25

#read data
with open(df_csv_file, mode = 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

#read the dict of word information gain
with open(word_info_file, 'rb') as handle:
  word_info = pickle.loads(handle.read())

def_word_info = defaultdict(np.float64,word_info)
#print(type(def_word_info['joaqamole']))
#print(type(def_word_info['a']))
#print(word_info['joaqamole'])


def word_imp(word):  #a is a word (string)
    imp = def_word_info[word]
    return imp

#word = 'joaqamole'
#print(word_imp(word))


model = SentenceTransformer('all-MiniLM-L6-v2')
#text1_emb = model.encode(text1)



#for i in range(5002):
for i in range(50):
    sample_label = data[i][0]
    class_label = data[i][1] 
#    print(sample_label, class_number[class_label])
#    print(class_number[class_label])\
    
    parts = [x.strip() for x in sample_label.split("_")]
    (uid,imgid) = parts

    doc = get_profile_words_list(uid,imgid)
#    print(doc,'\n')
    
#    list1.sort(key=sortSecond)
#    sorted(L, key=len))
    sorted_doc = sorted(doc, reverse = True, key = word_imp)
#    print(sorted_doc,'\n')
    
#    take only top profile_average_words words
    topn_sorted_doc = sorted_doc[:profile_average_words]
#    print(topn_sorted_doc,'\n')
    
    encoded_doc = np.zeros((0,384),dtype = np.single)
    
    for i in range(len(topn_sorted_doc)):
#        print(topn_sorted_doc[i])
#        break
        i_emb = model.encode(topn_sorted_doc[i])
        encoded_doc = np.vstack((encoded_doc,i_emb))
    
    print('before padding size ',encoded_doc.shape)
    
    padsize = profile_average_words - len(topn_sorted_doc)
    if( padsize > 0):
        pad_array = np.zeros((padsize,384),dtype = np.single)
        print(pad_array.shape)
        encoded_doc = np.vstack((encoded_doc,pad_array))
    
    print(encoded_doc.shape, encoded_doc[14][0] ,'\n')
    
#    problem if it does not have average number of words
#    solution padding. loop as many words are there, then vstack the remaining size
    
#    maybe return encoded_doc
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    res = sorted(test_list, key = lambda ele: test_dict[ele])
#    sorted_doc = sorted(doc, key = lambda ele: word_info[ele])
#    print(sorted_doc,'\n')
