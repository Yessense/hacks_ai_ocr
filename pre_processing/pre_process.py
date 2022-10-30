import base64
from urllib import request

import cv2
import flask
import numpy as np


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
        img = self.remove_noise(img)
        img = self.thresholding(img)
        return img


preproc = Preprocessing()
app = flask.Flask(__name__)


@app.route('/preprocess_img', methods=['POST'])
def post():

    img_bytes = request.json['img']
    img = base64.decodebytes(img_bytes.encode('utf-8'))
    img = cv2.imdecode(np.frombuffer(img, np.uint8), -1)

    result = preproc.preprocessing(img)

    # plt.imshow(result)
    # plt.show()
    result = cv2.imencode('.jpg', result)[1].tostring()
    data = {}
    data['img'] = base64.encodebytes(result).decode('utf-8')

    return str(data)


app.run(host='0.0.0.0', port=8084)
