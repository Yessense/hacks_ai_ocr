import csv
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from dataset import WordsinBoxes
import requests
import json
import base64
import matplotlib.pyplot as plt
import numpy as np
import cv2
from metrics import sber_metric
from spell_checker import post_processing
from skimage import io

test_dir = Path('/home/yessense/data/hacks_ai_ocr/test_dataset/')
labels = test_dir / 'test.csv'

df = pd.read_csv(labels)
print(df.info())
print(df.head())


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
    try:
        bounds = requests.post("http://127.0.0.1:8080/get_bounds", json=data)
        output = bounds.json()
    except json.decoder.JSONDecodeError:
        return ""
    if len(output):
        text = []
        for answer in output:
            text.append(answer[1])
        text = "\n".join(text)
    else:
        text = ""
    return text


for index, row in tqdm(df.iterrows()):

    path_to_img = test_dir / row['image_path']

    image = io.imread(path_to_img)
    output = eval(row['output'])

    box: dict
    for box in output[0]:
        l, t, w, h = [box[key] for key in ['left', 'top', 'width', 'height']]
        height, width = int(image.shape[0]), int(image.shape[1])

        if isinstance(l, float):
            l = width * l
            w = width * w
            t = height * t
            h = height * h
        elif isinstance(l, int):
            pass
        else:
            raise TypeError("l is wrong type")

        l, w, t, h = map(int, [l, w, t, h])
        part_of_image = image[t:t + h + 1, l: l + w + 1]

        if part_of_image.shape[0] == 0 or part_of_image.shape[1] == 0:
            label = ""
        else:
            label = get_answer(part_of_image)
        # plt.imshow(image)
        # plt.show()
        #
        # plt.imshow(part_of_image)
        # plt.show()

        if label == ''
        box['label'] = label
        label.replace('"', '')

    df['output'][index] = output
    # print(row['output'])





df.to_csv("Герои ML и магии.csv", index=False)
