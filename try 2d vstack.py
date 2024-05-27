import numpy as np


a = np.array([[1], [2], [3]])
b = np.array([[4], [5], [6]])

c = np.vstack((a,b))

#print(c)


image_ocr1 = np.load(r'E:\CVPR 23\Datasets\Our Dataset 5 Class\temp\O\10234782_25\10234782_25_image_ocr.npy')
image_ocr2 = np.load(r'E:\CVPR 23\Datasets\Our Dataset 5 Class\temp\N\10773912_26\10773912_26_image_ocr.npy')

stack = np.vstack((image_ocr1,image_ocr2))

#print(image_ocr1.shape)
#print(image_ocr2.shape)
#print(stack.shape)



array = np.zeros((0,384),dtype = np.single)
print(array.shape)
array = np.vstack((array,image_ocr1))
print(array.shape)
array = np.vstack((array,image_ocr2))
print(array.shape)

#array = np.delete(array,(1,384))














