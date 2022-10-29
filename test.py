import json

import easyocr

import PIL
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

reader = easyocr.Reader(['ru', 'en'], gpu=True)


def ocr_image(img, max_info=True):
    number = '0123456789'
    symbol = "!\"#$%&'()*+,-./:;<=>?@[\\]№_`{|} €₽"
    lang_char = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    used_symbols = number + symbol + lang_char
    if max_info:
        out = reader.readtext(np.array(img), paragraph=False, detail=1, allowlist=used_symbols)
    else:
        out = reader.readtext(np.array(img), paragraph=True, allowlist=used_symbols)

    return out


def draw_boxes(image, bounds, color='yellow', width=2):
    # fontpath = "./CYRIL1.TTF"
    # font = ImageFont.truetype(fontpath, 32)

    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        text = bound[1]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
        # draw.text(p0, text, font=font, fill=(255, 0, 0))

    return image


def process_image(path_to_image, show_max=False):
    image = Image.open(path_to_image)

    bounds = ocr_image(image, max_info=show_max)

    image = draw_boxes(image, bounds)

    plt.imshow(image)
    plt.show()

    return bounds


bounds = process_image('/home/yessense/data/hacks_ai_ocr/dataset/train/10622.jpg')
bounds2 = process_image('/home/yessense/data/hacks_ai_ocr/dataset/train/10622.jpg', show_max=True)
# bounds = process_image('/home/yessense/data/hacks_ai_ocr/dataset/train/00012.jpg')
# bounds2 = process_image('/home/yessense/data/hacks_ai_ocr/dataset/train/00012.jpg', show_max=True)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

jsn = json.dumps(bounds2, cls=NpEncoder)
print(jsn)

# 'train/00012.jpg,"[[{""left"": 0.009510869565217392, ""top"": 0.19429347826086957, ""width"": 0.4293478260869565, ""height"": 0.3383152173913044, ""label"": ""Мангальные \\/n блюда \\/n шашлык \\/n баранина говядина \\/n курица сёмга \\/n домашняя свинина"", ""shape"": ""rectangle""}, {""left"": 0.014945652173913044, ""top"": 0.5475543478260869, ""width"": 0.4470108695652174, ""height"": 0.29891304347826086, ""label"": ""печень \\/n домашняя в сетке \\/n цыплёнок \\/n табака"", ""shape"": ""rectangle""}, {""left"": 0.4592391304347826, ""top"": 0.19021739130434784, ""width"": 0.26086956521739135, ""height"": 0.25407608695652173, ""label"": ""хачапури \\/n под \\/n заказ"", ""shape"": ""rectangle""}, {""left"": 0.7296195652173914, ""top"": 0.19701086956521738, ""width"": 0.21875, ""height"": 0.2459239130434783, ""label"": ""пиво \\/n на \\/n розлив"", ""shape"": ""rectangle""}, {""left"": 0.4741847826086957, ""top"": 0.44972826086956524, ""width"": 0.33016304347826086, ""height"": 0.15081521739130427, ""label"": ""пельмени \\/n домашние \\/n на вынос"", ""shape"": ""rectangle""}, {""left"": 0.47554347826086957, ""top"": 0.6073369565217391, ""width"": 0.23505434782608697, ""height"": 0.21059782608695654, ""label"": ""кофе \\/n  с \\/n собой"", ""shape"": ""rectangle""}, {""left"": 0.7350543478260869, ""top"": 0.6127717391304348, ""width"": 0.19836956521739135, ""height"": 0.21195652173913038, ""label"": ""бизнес \\/n ланч \\/n C12 00 - 15 00"", ""shape"": ""rectangle""}]]"
print("DOne")
