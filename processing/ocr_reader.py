import base64
import json
from argparse import ArgumentParser
import numpy as np

import easyocr
import cv2
from flask import Flask, request

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
    parser.add_argument('--paragraph', type=bool, default=True)
    parser.add_argument('--allowed_list', type=str, default=NUMBERS + SYMBOLS + LANG_CHARS)
    parser.add_argument('--y_ths', type=float, default=0.2)
    args = parser.parse_args()

    # Model
    reader = easyocr.Reader(['ru', 'en'], gpu=True, download_enabled=False,
                            model_storage_directory='/home/.EasyOCR/model')

    # Flask logic
    app = Flask(__name__)


    @app.route('/get_bounds', methods=['POST'])
    def get_embeddings():
        """
        {
            'img': [...]
        }
        :return:
        [
            [
                [
                    [x1, y1]
                    [x2, y2]
                    [x3, y3]
                    [x4, y4]
                ], 'text'
            ]
        ]
        """
        img_bytes = request.json['img']
        img = base64.decodebytes(img_bytes.encode('utf-8'))
        img = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bounds = reader.readtext(img, paragraph=args.paragraph, detail=args.detail,
                                 allowlist=args.allowed_list, y_ths=args.y_ths)

        out = json.dumps(bounds, cls=NpEncoder)
        return str(out)


    print('run')
    app.run(host='0.0.0.0', port=8080)
