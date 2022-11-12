import base64

import cv2
from flask import Flask, request, jsonify
import numpy as np


class Preprocessing:
    def get_grayscale(self, image):
        """ Grayscale transformation"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise(self, image):
        """ Noise removal"""
        return cv2.medianBlur(image, 5)

    def thresholding(self, image):
        """ Thresholding"""
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def dilate(self, image):
        """ Dilation"""
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    def erode(self, image):
        """ Erosion"""
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    def opening(self, image):
        """ Opening - erosion followed by dilation"""
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def canny(self, image):
        """ Canny edge detection"""
        return cv2.Canny(image, 100, 200)

    def deskew(self, image):
        """ skew correction"""
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
        transformations = [
            self.get_grayscale,
            self.remove_noise,
            self.thresholding,
            # self.deskew,
            # self.canny,
            # self.opening,
        ]
        for transformation in transformations:
            img = transformation(img)
        return img


preproc = Preprocessing()
app = Flask(__name__)


@app.route('/preprocess_img', methods=['POST'])
def post():
    """
    {
        'img': [...]
    }
    :return:
    """
    img_bytes = request.json['img']
    img = base64.decodebytes(img_bytes.encode('utf-8'))
    img = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = preproc.preprocessing(img)

    result = cv2.imencode('.jpg', result)[1].tostring()
    data = {}
    data['img'] = base64.encodebytes(result).decode('utf-8')
    answer = jsonify(data)
    print(answer)

    return answer


app.run(host='0.0.0.0', port=8084)
