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
bounds = requests.post("http://127.0.0.1:8080/get_bounds", json=data)
print(bounds.json())

# Decode image if needs
jsn = json.dumps(data)

out = json.loads(jsn)
new_img = out['img']
new_img = new_img.encode('utf-8')
new_img = base64.decodebytes(new_img)

image = cv2.imdecode(np.frombuffer(new_img, np.uint8), -1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show image
plt.imshow(image)
plt.show()
