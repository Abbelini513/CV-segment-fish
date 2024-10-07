# Выполнила Бондарева Алина Кирилловна
import cv2
import glob
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from utils.compute_iou import compute_ious


def segment_fish(img):
    """
    This method should compute masks for given image
    Params:
        img (np.ndarray): input image in BGR format
    Returns:
        mask (np.ndarray): fish mask. should contain bool values
    """
    # Преобразуем изображение в HSV-формат
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Диапазоны для оранжевого и белого цветов
    light_orange = (1, 190, 150)
    dark_orange = (30, 255, 255)
    light_white = (60, 0, 200)
    dark_white = (145, 150, 255)

    # Маски для цветов
    mask_orange = cv2.inRange(img_hsv, light_orange, dark_orange)
    mask_white = cv2.inRange(img_hsv, light_white, dark_white)

    # Делаем доп. маску для темных областей
    dark_shadow = (0, 0, 50)
    bright_shadow = (180, 255, 120)
    mask_shadow = cv2.inRange(img_hsv, dark_shadow, bright_shadow)

    # Добавляем адаптивную фильтрацию по яркости (V-канал)
    v_channel = img_hsv[:, :, 2]
    _, mask_brightness = cv2.threshold(v_channel, 120, 255, cv2.THRESH_BINARY)

    # Объединение масок
    mask_fish = cv2.bitwise_or(mask_orange, mask_white)
    mask_fish = cv2.bitwise_or(mask_fish, mask_shadow)
    mask_fish = cv2.bitwise_and(mask_fish, mask_brightness)

    # Применение морфологических операций
    kernel = np.ones((7, 7), np.uint8)
    mask_fish = cv2.morphologyEx(mask_fish, cv2.MORPH_OPEN, kernel)
    mask_fish = cv2.morphologyEx(mask_fish, cv2.MORPH_CLOSE, kernel)

    # Применяем градиент Собеля для улучшения контуров
    sobel_x = cv2.Sobel(mask_fish, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(mask_fish, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.sqrt(sobel_x ** 2 + sobel_y ** 2)
    _, mask_gradient = cv2.threshold(sobel_combined, 60, 255, cv2.THRESH_BINARY)

    # Добавляем морфологический градиент для контуров
    mask_contour = cv2.morphologyEx(mask_fish, cv2.MORPH_GRADIENT, kernel)
    mask_fish = cv2.bitwise_or(mask_fish, mask_contour)
    mask_fish = cv2.bitwise_or(mask_fish, mask_gradient.astype(np.uint8))

    mask_fish_bool = mask_fish.astype(bool)

    return mask_fish_bool


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--is_train", action="store_true")
    args = parser.parse_args()
    stage = 'train' if args.is_train else 'test'

    data_root = osp.join("dataset", stage, "imgs")
    img_paths = glob.glob(osp.join(data_root, "*.jpg"))
    len(img_paths)

    masks = dict()
    for path in img_paths:
        img = cv2.imread(path)
        mask = segment_fish(img)
        masks[osp.basename(path)] = mask
    print(compute_ious(masks, osp.join("dataset", stage, "masks")))
