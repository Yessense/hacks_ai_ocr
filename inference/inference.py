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


class Preprocessing:
    # get grayscale image
    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    def remove_noise(self, image):
        return cv2.medianBlur(image, 5)

    # thresholding
    def thresholding(self, image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # dilation
    def dilate(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    # erosion
    def erode(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    # opening - erosion followed by dilation
    def opening(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # canny edge detection
    def canny(self, image):
        return cv2.Canny(image, 100, 200)

    # skew correction
    def deskew(self, image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def preprocessing(self, img):
        img = self.get_grayscale(img)
        # img = self.remove_noise(img)
        # img = self.thresholding(img)
        return img


preproc = Preprocessing()


metric = 0
for i, batch in enumerate(wib):
    image, label = batch
    image = preproc.preprocessing(image)
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
