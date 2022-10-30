from dataset import WordsinBoxes
import requests
import json
import base64
import matplotlib.pyplot as plt
import numpy as np
import cv2
from metrics import sber_metric
from spell_checker import post_processing

wib = WordsinBoxes(csv_file="../dataset/train.csv", root_dir="../dataset/", n_images=None,
                   cached_info_path='./cache.npz')

def get_answer(image):
    for dim in image.shape:
        if dim == 0:
            return []
    img_str = cv2.imencode('.jpg', image)[1].tobytes()
    data = {}
    data['img'] = base64.encodebytes(img_str).decode('utf-8')

    jsn = json.dumps(data)

    out = json.loads(jsn)
    new_img = out['img']
    new_img = new_img.encode('utf-8')
    new_img = base64.decodebytes(new_img)

    image = cv2.imdecode(np.frombuffer(new_img, np.uint8), -1)
    # plt.imshow(image)
    # plt.show()
    bounds = requests.post("http://127.0.0.1:8080/get_bounds", json=data)

    return bounds.json()


metric = 0
for i, batch in enumerate(wib):
    image, label = batch
    output = get_answer(image)
    if len(output):
        text = []
        for answer in output:
            text.append(answer[1])
        # text = post_processing(text)
        text = "\n".join(text)
        current_metric = sber_metric(text, label)
        metric += current_metric
    else:
        current_metric = 0.
    if i % 100 == 0:
        print(f'{i}. current: {current_metric}, total: {metric/(i+1):0.4f} ')

metric /= len(wib)


print(metric)
