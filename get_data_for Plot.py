import numpy as np
import os
from pathlib import Path




FOLDER = r'E:\CVPR 23\Datasets\Our Dataset 5 Class\temp'
class_list = os.listdir(FOLDER)

class_dict_image_ocr = {0:r'E:\CVPR 23\Datasets\Our Dataset 5 Class\A_image_ocr.npy',
                        1:r'E:\CVPR 23\Datasets\Our Dataset 5 Class\C_image_ocr.npy',
                        2:r'E:\CVPR 23\Datasets\Our Dataset 5 Class\E_image_ocr.npy',
                        3:r'E:\CVPR 23\Datasets\Our Dataset 5 Class\N_image_ocr.npy',
                        4:r'E:\CVPR 23\Datasets\Our Dataset 5 Class\O_image_ocr.npy'}

class_dict_profile_desc = {0:r'E:\CVPR 23\Datasets\Our Dataset 5 Class\A_profile_desc.npy',
                        1:r'E:\CVPR 23\Datasets\Our Dataset 5 Class\C_profile_desc.npy',
                        2:r'E:\CVPR 23\Datasets\Our Dataset 5 Class\E_profile_desc.npy',
                        3:r'E:\CVPR 23\Datasets\Our Dataset 5 Class\N_profile_desc.npy',
                        4:r'E:\CVPR 23\Datasets\Our Dataset 5 Class\O_profile_desc.npy'}


part1 = '_image_ocr'
part2 = '_dp_ocr'
part3 = '_banner_ocr'

part4 = '_image_labels'
part5 = '_dp_labels'
part6 = '_banner_labels'

part7 = '_image_desc'
part8 = '_profile_desc'
  
#print("classes are '", path, "' :") 
  
# print the list
#print(class_list)

# =============================================================================
# for i in range(5):
# #    print(i)
#     print(class_list[i])
#     
#     class_label = class_list[i]
#     class_path = Path(FOLDER,class_label)
#     samples = os.listdir(class_path)
# #    print(len(samples))
#     
#     ocr_collection = np.zeros((0,384),dtype = np.single)
#     
#     for j in range(len(samples)):
# #        print(i)
#         sample = samples[j]
# #        print(sample)
#         
#         filename = sample + part1 + '.npy' 
# #        print(filename)
#         filepath = Path(class_path,sample,filename)
# #        print(filepath)
#         
#         
# #        image_ocr = 2093301_21_image_ocr.npy
#         image_ocr = np.load(filepath)
# #        print(image_ocr.shape)
#         
#         
#         if(image_ocr.size != 0):
#             ocr_collection = np.vstack((ocr_collection,image_ocr))
# #            print(ocr_collection.shape)
#         
#         print(j)
#     
#     print(ocr_collection.shape)    
#     np.save(class_dict_image_ocr[i], ocr_collection)
# =============================================================================



for i in range(5):
#    print(i)
    print(class_list[i])
    
    class_label = class_list[i]
    class_path = Path(FOLDER,class_label)
    samples = os.listdir(class_path)
#    print(len(samples))
    
    profile_desc_collection = np.zeros((0,384),dtype = np.single)
    
    for j in range(len(samples)):
#        print(i)
        sample = samples[j]
#        print(sample)
        
        filename = sample + part8 + '.npy' 
#        print(filename)
        filepath = Path(class_path,sample,filename)
#        print(filepath)
        
        
#        image_ocr = 2093301_21_image_ocr.npy
        profile_desc = np.load(filepath)
#        print(image_ocr.shape)
        
        
        if(profile_desc.size != 0):
            profile_desc_collection = np.vstack((profile_desc_collection,profile_desc))
#            print(ocr_collection.shape)
        
        print(j)
    
    print(profile_desc_collection.shape)    
    np.save(class_dict_profile_desc[i], profile_desc_collection)       








    
    
    
    
    
    
    
    
    
    
    
#    new_dir = Path(FOLDER,class_label,directory)
    
    

