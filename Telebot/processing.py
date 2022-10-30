import io

import pyttsx3
import time

import requests
import base64
import numpy as np
import cv2
from pathlib import Path

from io import BytesIO
from PIL import Image


class TextToVoice:
    def __init__(self):
        self.__model = pyttsx3.init()
        self.__model.setProperty('voice', 'russian')

    def get_voice_rb(self, text):
        path_to_file = Path("files/last_voice.oga")
        self.__model.save_to_file(text, path_to_file)
        self.__model.runAndWait()

        print("waiting started")

        for i in range(100):
            time.sleep(1)
            print(i)
            if path_to_file.is_file():
                return open("files/last_voice.oga", 'rb')
        print("waiting failed...")
        return None


class ImgToText:
    def __init__(self):
        pass

    @staticmethod
    def get_bounds(image):
        data = {}
        data['img'] = base64.encodebytes(image).decode('utf-8')

        bounds = requests.post("http://192.168.50.84:8080/get_bounds", json=data)

        return bounds.json()


class Visualize:
    def __init__(self):
        pass

    @staticmethod
    def put_box(image, points: list, label: str):
        label = label.replace('ё', 'е')
        (tl, tr, br, bl) = points
        tl = np.array((int(tl[0]), int(tl[1])))
        tr = np.array((int(tr[0]), int(tr[1])))
        br = np.array((int(br[0]), int(br[1])))
        bl = np.array((int(bl[0]), int(bl[1])))

        cv2.rectangle(image, tl, br, (0, 255, 0), 1)
        a = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                                       (bl - tl)[1])
        cv2.rectangle(image, tl + np.array([0, -int(a * 10)]),
                      tl + [int(len(label) * a * 7), 0],
                      (0, 255, 0), -1)
        cv2.putText(image,
                    label,
                    tl,
                    cv2.FONT_HERSHEY_COMPLEX,
                    a / 3, (0, 0, 0), 1)

        return image

    @staticmethod
    def get_visualisation(image, bounds):
        pil_image = Image.open(BytesIO(image))
        image_tensor = np.asarray(pil_image)

        for bound in bounds:
            image_tensor = Visualize.put_box(image_tensor, bound[0], bound[1])

        image = Image.fromarray(image_tensor, 'RGB')

        image_bytes = io.BytesIO()
        image.save(image_bytes, format=pil_image.format)
        return image_bytes.getvalue()


class SpellChecker:
    def __init__(self):
        pass

    @staticmethod
    def check_spelling(bounds):

        # return bounds
        last = -2 if len(bounds) & len(bounds[0]) == 3 else -1

        text = {
            "box": list([i[last] for i in bounds])
        }

        got_bounds = requests.post("http://192.168.50.39:1707/get_spell_check", json = text)
        print(got_bounds.json())
        for i, b in enumerate(got_bounds.json()['result']):
            bounds[i][last] = b

        return bounds
