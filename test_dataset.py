import csv
import json
import matplotlib.pyplot as plt

import cv2


def get_box(num: int):
    path = output = ''
    with open('./dataset/train.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            path = row['image_path']
            output = json.loads(row['output'])[0]
            if str(num) in path:
                break
    img = cv2.imread('./dataset/'+path)
    h, w, _ = img.shape
    #print(h, w)
    # проход по всем боксам
    for box in output:
        if box['height'] > 1:
            box['height'] /= h
        if box['width'] > 1:
            box['width'] /= w
        if box['left'] > 1:
            box['left'] /= w
        if box['top'] > 1:
            box['top'] /= h
        for i in ['left', 'top', 'height', 'top']:
            if box[i] < 0:
                box[i] = 0
        tl = (int(box['left']*w), int(box['top']*h))
        br = (int((box['left']+box['width'])*w), int((box['top']+box['height'])*h))
        #print(tl, tr, bl, br)
        cv2.rectangle(img, tl, br, (0, 255, 0), 2)
        cv2.putText(img,
                    box['label'],
                    (tl[0], tl[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5, (0, 255, 0), 2)
    return img


if __name__ == '__main__':
    img = get_box(37316)
    plt.imshow(img)
    plt.show()
    print("Done")