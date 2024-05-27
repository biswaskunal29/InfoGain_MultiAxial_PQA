import numpy as np

# =============================================================================
#     for text inside image
# =============================================================================
#names of files
class_o_image_ocr = 'O_image_ocr.npy'
class_c_image_ocr = 'C_image_ocr.npy'
class_e_image_ocr = 'E_image_ocr.npy'
class_a_image_ocr = 'A_image_ocr.npy'
class_n_image_ocr = 'N_image_ocr.npy'


#loading the numpy arrays
o_array_image_ocr = np.load(class_o_image_ocr)
c_array_image_ocr = np.load(class_c_image_ocr)
e_array_image_ocr = np.load(class_e_image_ocr)
a_array_image_ocr = np.load(class_a_image_ocr)
n_array_image_ocr = np.load(class_n_image_ocr)


#checking shape
print(o_array_image_ocr.shape,"\tclass O images have ",o_array_image_ocr.shape[0], ' words')
print(c_array_image_ocr.shape,"\tclass C images have ",c_array_image_ocr.shape[0], ' words')
print(e_array_image_ocr.shape,"\tclass E images have ",e_array_image_ocr.shape[0], ' words')
print(a_array_image_ocr.shape,"\tclass A images have ",a_array_image_ocr.shape[0], ' words')
print(n_array_image_ocr.shape,"\tclass N images have ",n_array_image_ocr.shape[0], ' words')

print('\n')
#if we want to take only first n words n = 10 (say)
first_n = 10
print(o_array_image_ocr[:first_n].shape)
print(c_array_image_ocr[:first_n].shape)
print(e_array_image_ocr[:first_n].shape)
print(a_array_image_ocr[:first_n].shape)
print(n_array_image_ocr[:first_n].shape)
print('\n')
# =============================================================================
# plot to check gaussian of not
# =============================================================================





# =============================================================================
#     for profile description text
# =============================================================================
#names of files
class_o_profile_desc = 'O_profile_desc.npy'
class_c_profile_desc = 'C_profile_desc.npy'
class_e_profile_desc = 'E_profile_desc.npy'
class_a_profile_desc = 'A_profile_desc.npy'
class_n_profile_desc = 'N_profile_desc.npy'


#loading the numpy arrays
o_array_profile_desc = np.load(class_o_profile_desc)
c_array_profile_desc = np.load(class_c_profile_desc)
e_array_profile_desc = np.load(class_e_profile_desc)
a_array_profile_desc = np.load(class_a_profile_desc)
n_array_profile_desc = np.load(class_n_profile_desc)


#checking shape
print(o_array_profile_desc.shape,"\tclass O profile description have ",o_array_profile_desc.shape[0], ' words')
print(c_array_profile_desc.shape,"\tclass C profile description have ",c_array_profile_desc.shape[0], ' words')
print(e_array_profile_desc.shape,"\tclass E profile description have ",e_array_profile_desc.shape[0], ' words')
print(a_array_profile_desc.shape,"\tclass A profile description have ",a_array_profile_desc.shape[0], ' words')
print(n_array_profile_desc.shape,"\tclass N profile description have ",n_array_profile_desc.shape[0], ' words')

print('\n')
#if we want to take only first n words n = 10 (say)
first_n = 10
print(o_array_profile_desc[:first_n].shape)
print(c_array_profile_desc[:first_n].shape)
print(e_array_profile_desc[:first_n].shape)
print(a_array_profile_desc[:first_n].shape)
print(n_array_profile_desc[:first_n].shape)

# =============================================================================
# plot to check gaussian of not
# =============================================================================
















