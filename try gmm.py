from sklearn.mixture import GaussianMixture
import numpy as np



class_o_image_ocr = 'O_image_ocr.npy'
o_array_image_ocr = np.load(class_o_image_ocr)
print(o_array_image_ocr.shape,"\tclass O images have ",o_array_image_ocr.shape[0], ' words')

gmm = GaussianMixture(n_components = 3)
gmm.fit(o_array_image_ocr)

# Assign a label to each sample
labels = gmm.predict(o_array_image_ocr)
#d['labels']= labels
#d0 = d[d['labels']== 0]
#d1 = d[d['labels']== 1]
#d2 = d[d['labels']== 2]
# 
## plot three clusters in same plot
#plt.scatter(d0[0], d0[1], c ='r')
#plt.scatter(d1[0], d1[1], c ='yellow')
#plt.scatter(d2[0], d2[1], c ='g')

































