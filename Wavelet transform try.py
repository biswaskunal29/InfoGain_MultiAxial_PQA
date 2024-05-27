import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import cv2

import pywt
import pywt.data


# Load image
#original = pywt.data.camera()
original = Image.open("small_circle.jpg").convert('L')
#original = plt.imread("open.jpg")
#original = original.convert("L")



# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3', axes=(-2, -1))
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()

#LL.save("LL.jpg")

#im = Image.fromarray(LL)
#im.save("LL.jpeg")


#destRGB = cv2.cvtColor(LL, cv2.COLOR_BGR2RGB)
#matplotlib.image.imsave('LL.png', destRGB)
#
#plt.imsave('Neur L2 HH LL.png', LL, cmap='gray')
#plt.imsave('Neur L2 HH LH.png', LH, cmap='gray')
#plt.imsave('Neur L2 HH HL.png', HL, cmap='gray')
#plt.imsave('Neur L2 HH HH.png', HH, cmap='gray')




















