import json
import base64
import matplotlib.pyplot as plt
import numpy as np
import cv2

path = '/home/yessense/data/hacks_ai_ocr/dataset/train/00001.jpg'
data = {}
with open(path, mode='rb') as file:
    img = file.read()
data['img'] = base64.encodebytes(img).decode('utf-8')

jsn = json.dumps(data)

out = json.loads(jsn)
new_img = out['img']
new_img = new_img.encode('utf-8')
new_img = base64.decodebytes(new_img)

image = cv2.imdecode(np.frombuffer(new_img, np.uint8), -1)
plt.imshow(image)
plt.show()


print("Done")