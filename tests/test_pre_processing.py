import requests
import json
import base64
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Open image
path_to_image = './images/37316.jpg'
with open(path_to_image, mode='rb') as file:
    img = file.read()

# Encode image
data = {}
data['img'] = base64.encodebytes(img).decode('utf-8')

# Get answer from processing container
response = requests.post("http://127.0.0.1:8084/preprocess_img", json=data)
image_json = response.json()
# out = json.loads(img_bytes.json())
new_img = image_json['img']
new_img = new_img.encode('utf-8')
new_img = base64.decodebytes(new_img)

image = cv2.imdecode(np.frombuffer(new_img, np.uint8), -1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show image
plt.imshow(image)
plt.show()
