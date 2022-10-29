from pathlib import Path

from torchvision.datasets.vision import VisionDataset  # , DatasetFolder
from skimage import io
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os
from collections import Counter

import cv2
import numpy as np


# img, [[[671, 484], [890, 484], [890, 569], [671, 569]]
def cut_img(img, points: list):
    """
    img -- изображение
    points: list -- координаты углов бокса
    """
    (tl, bl, br, tr) = points
    tl = np.array((int(tl[0]), int(tl[1])))
    tr = np.array((int(tr[0]), int(tr[1])))
    br = np.array((int(br[0]), int(br[1])))
    bl = np.array((int(bl[0]), int(bl[1])))

    # Define bounding boxes
    h, w, _ = img.shape
    if tl[0] < 1: tl *= h
    if bl[0] < 1: bl *= h
    if br[0] < 1: br *= h
    if tr[0] < 1: tr *= h

    new_img = img[tl[0]:bl[0], tl[1]:tr[1]]
    return new_img


def count_boxes(df, n_images=-2):
    col_names = ["image_path", "output"]
    count = 0

    for i, row in df.iterrows():

        if i == n_images:  # !
            break

        image_path = row["image_path"]
        boxes = eval(row["output"])[0]
        num_of_boxes = len(boxes)

        for j, box in enumerate(boxes):
            count += 1

    total_len = count
    count = -1

    img_arr = np.zeros(total_len, dtype=np.uint)
    box_arr = np.zeros(total_len, dtype=np.uint)

    for i, row in df.iterrows():

        if i == n_images:  # !
            break

        image_path = row["image_path"]
        boxes = eval(row["output"])[0]
        num_of_boxes = len(boxes)

        for j, box in enumerate(boxes):
            count += 1
            img_arr[count] = i
            box_arr[count] = j

    return total_len, img_arr, box_arr


class WordsinBoxes(Dataset):

    def __init__(self, csv_file, root_dir, n_images=5, cached_info_path='cache.npz'):
        self.n_images = n_images
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

        cached_info_path = Path(cached_info_path)
        # file exists
        if cached_info_path.is_file():
            npz_file = np.load(cached_info_path)
            self.total_len, self.img_arr, self.box_arr = [npz_file[key] for key in npz_file.files]
        else:
            self.total_len, self.img_arr, self.box_arr = count_boxes(self.frame, self.n_images)
            np.savez(cached_info_path, total_len=self.total_len, img_arr=self.img_arr, box_arr=self.box_arr)

        self.alph = Counter()

    def __len__(self):
        return self.total_len

    def find_idx(self, idx):
        img_idx = self.img_arr[idx]
        box_idx = self.box_arr[idx]
        return img_idx, box_idx

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_idx, box_idx = self.find_idx(idx)
        img_name = self.frame.iloc[image_idx][0]
        img_path = os.path.join(self.root_dir, img_name)
        image = io.imread(img_path)

        boxes = eval(self.frame.iloc[image_idx][1])[0]
        box = boxes[box_idx]

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
        image = image[t:t + h + 1, l: l + w + 1]
        label = box['label']

        return image, label

    def show(self, n=10):

        fig = plt.figure(figsize=(30, n // 2 * 10))

        for i in range(len(self)):
            image, label = self[i]

            ax = plt.subplot(n, 1, i + 1)
            ax.set_title(f'Sample №{i}:\n{label}')
            ax.axis('off')
            plt.imshow(image)

            if i == n - 1:
                plt.show()
                break

    def find_alph(self):
        for i in range(len(self)):
            _, label = self[i]
            self.alph.update(list(label))

    def analyze_alph(self):
        total = sum(self.alph.values())
        print('Всего символов:', total)
        print('Всего разных символов:', len(self.alph))
        c = self.alph.most_common()
        print(c)

        p = list(map(lambda x: [x[0],
                                float("{:.2e}".format(x[1] / total))
                                ], c))
        print(p)


if __name__ == '__main__':
    wib = WordsinBoxes(csv_file="../dataset/train.csv", root_dir="../dataset/", n_images=None,
                       cached_info_path='./cache.npz')
    image, label = wib[1]
    print(len(wib))
    wib.show(10)

    metric = 0

