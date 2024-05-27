from PIL import Image
import requests
from transformers import AutoProcessor, BlipModel

model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#image = Image.open(requests.get(url, stream=True).raw)

image = Image.open("Original_Image.jpg") 

inputs = processor(images=image, return_tensors="pt")

image_features = model.get_image_features(**inputs)

print(image_features.shape)











