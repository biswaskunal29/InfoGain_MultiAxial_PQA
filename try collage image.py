# importing the modules
import cv2
import numpy as np

# read all the images
# we are going to take 4 images only
image1=cv2.imread("Messi.jpg")
image2=cv2.imread("open.jpg")
image3=cv2.imread("red.jpg")
image4=cv2.imread("Original_Image.jpg")

# make all the images of same size 
#so we will use resize function
image1=cv2.resize(image1,(256,256))
image2=cv2.resize(image2,(256,256))
image3=cv2.resize(image3,(256,256))
image4=cv2.resize(image4,(256,256))

# Now how we will attach image with other image
# we will create a horizontal stack of images
# then we will add it to the vertical stack
# let the horizontal pair be (image1,image2)
# and (image3,image4)
# we will use numpy stack function
Horizontal1=np.hstack([image1,image2])
Horizontal2=np.hstack([image3,image4])

# Now the horizontal attachment is done
# noe vertical attachment
Vertical_attachment=np.vstack([Horizontal1,Horizontal2])

# Show the final attachment
cv2.imshow("Final Collage",Vertical_attachment)
#cv2.waitKey(0)
#cv2.destroyAllWindows()









































