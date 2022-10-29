import base64
import json
from argparse import ArgumentParser
import numpy as np

import easyocr
import cv2
# from PIL import ImageDraw
# from PIL import Image
from flask import Flask, request

# from matplotlib import pyplot as plt

NUMBERS = '0123456789'
SYMBOLS = "!\"#$%&'()*+,-./:;<=>?@[\\]№_`{|} €₽"
LANG_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


if __name__ == '__main__':
    # Parse args
    parser = ArgumentParser(description='Process text')
    parser.add_argument('--detail', type=int, default=1, choices=[0, 1])
    parser.add_argument('--paragraph', type=bool, default=False)
    parser.add_argument('--allowed_list', type=str, default=NUMBERS + SYMBOLS + LANG_CHARS)
    args = parser.parse_args()

    # Model
    reader = easyocr.Reader(['ru', 'en'], gpu=True, download_enabled=False, model_storage_directory='/home/.EasyOCR/model')

    # Flask logic
    app = Flask(__name__)


    # def draw_boxes(image, bounds, color='yellow', width=2):
    #     # fontpath = "./CYRIL1.TTF"
    #     # font = ImageFont.truetype(fontpath, 32)
    #     image = Image.fromarray(image, 'RGB')
    #
    #     draw = ImageDraw.Draw(image)
    #     for bound in bounds:
    #         p0, p1, p2, p3 = bound[0]
    #         text = bound[1]
    #         draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    #         # draw.text(p0, text, font=font, fill=(255, 0, 0))
    #     return image

    @app.route('/get_bounds', methods=['POST'])
    def get_embeddings():
        """
        {
            'img': [...]
        }
        :return:
        """
        img_bytes = request.json['img']
        img = base64.decodebytes(img_bytes.encode('utf-8'))
        img = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
        # print(img)
        bounds = reader.readtext(img, paragraph=args.paragraph, detail=args.detail, allowlist=args.allowed_list)
        # img = draw_boxes(img, bounds)

        # plt.imshow(img)
        # plt.show()

        # print(bounds)
        out = json.dumps(bounds, cls=NpEncoder)
        # print(out)
        return str(out)


    print('run')
    app.run(host='0.0.0.0', port=8080)
