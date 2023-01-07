import glob
import os
import pandas as pd
from functools import partial
from multiprocessing import cpu_count, Pool
import cv2
from tqdm.auto import tqdm

HEIGHT, WIDTH = 1080, 1920


def resize_df(size):
    resized_height, resized_width = size
    df = pd.read_csv('../input/SoccerNet/tracking/folds.csv')
    df['x'] = df['x'] * resized_width / WIDTH
    df['y'] = df['y'] * resized_height / HEIGHT
    df['w'] = df['w'] * resized_width / WIDTH
    df['h'] = df['h'] * resized_height / HEIGHT
    df['cx'] = df['cx'] * resized_width / WIDTH
    df['cy'] = df['cy'] * resized_height / HEIGHT
    df.to_csv(f'../input/SoccerNet/tracking/folds_resized{resized_height}.csv', index=False)

def _resize_image(image_path, size):
    resized_height, resized_width = size
    save_path = image_path.replace('img1', f'img_resized{resized_height}')
    if os.path.exists(save_path):
        return
    img = cv2.imread(image_path)
    img = cv2.resize(img, (resized_width, resized_height))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

def resize_images(size):
    
    func = partial(_resize_image, size=size)

    image_files = sorted(glob.glob('../input/SoccerNet/tracking/SNMOT-*/img1/*.jpg'))
    pool = Pool(processes=cpu_count())
    with tqdm(total=len(image_files)) as t:
        for _ in pool.imap_unordered(func, image_files):
            t.update(1)


for size in ((288, 512), (360, 640)):
    resize_df(size)
    resize_images(size)
