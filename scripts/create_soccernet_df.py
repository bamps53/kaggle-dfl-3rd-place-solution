import glob
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DATA_DIR = '../input/SoccerNet/tracking/'

# convert from x,y,w,h to yolo format
def get_yolo_format_bbox(img_w, img_h, box):
    w = box[2]
    h = box[3]
    xc = box[0] + int(np.round(w/2))
    yc = box[1] + int(np.round(h/2))
    box = [xc/img_w, yc/img_h, w/img_w, h/img_h]
    box = [f"{i:.4g}" for i in box]
    return box

# get SoccerNet label info
def get_info(info):
    results = []
    for line in open(info):
        m = re.match('trackletID_(\d+)= (\S*).*', line.replace(';', ' '))
        if m:
            if m.group(2) not in label_dict:
                #print('bad label:', m.group(2))
                continue
            results.append([m.group(1), m.group(2)])
    return pd.DataFrame(results, columns=['id', 'lbl']).set_index('id').to_dict()['lbl']


all_dirs = sorted(glob.glob(f'{DATA_DIR}/SNMOT*'))

label_dict = {'ball': 0, 'player': 1, 'referee': 2, 'goalkeeper': 3, 'goalkeepers': 3}

all_dfs = []
for this_dir in all_dirs:
    video = this_dir.split('/')[-1]
    info = this_dir + '/gameinfo.ini'
    det = this_dir + '/gt/gt.txt'
    info_dict = get_info(info)
    det_df = pd.read_csv(det, names=['frame', 'player', 'x', 'y', 'w', 'h', 'f1', 'f2', 'f3', 'f4'], usecols=[
                         'frame', 'player', 'x', 'y', 'w', 'h'])
    det_df['label'] = det_df.player.astype(str).map(info_dict)
    det_df['label_id'] = det_df['label'].map(label_dict)
    det_df['video_id'] = video
    det_df = det_df.query('label == "ball"').reset_index(drop=True)
    det_df['count'] = det_df['player'].map(
        det_df.groupby('player').size().to_dict())
    all_dfs.append(det_df)
df = pd.concat(all_dfs).reset_index(drop=True)

df = df[~df[['frame', 'video_id']].duplicated(keep=False)]
df = df.sort_values(['video_id', 'frame']).reset_index(drop=True)


kf = GroupKFold(20)

df['folds'] = -1
for fold, (_, val_idx) in enumerate(kf.split(df, groups=df['video_id'])):
    df.loc[val_idx, 'folds'] = fold

df['cx'] = df['x'] + df['w'] / 2
df['cy'] = df['y'] + df['h'] / 2

df.to_csv('../input/SoccerNet/tracking/folds.csv', index=False)
