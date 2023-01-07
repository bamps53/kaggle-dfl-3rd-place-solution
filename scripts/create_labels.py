import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

event_attribute_classes = [
    'ball_action_forced',
    'opponent_dispossessed',
    'possession_retained',
    'fouled',
    'opponent_rounded',
    'challenge_during_ball_transfer',
    'pass__openplay',
    'cross__openplay',
    'pass__freekick',
    'cross__corner',
    'cross__freekick',
    'pass__corner',
    'pass',
    'cross'
]
event_classes = [
    'challenge',
    'play',
    'throwin'
]

def get_video_frames(video_name):
    video_dir = f'../input/train_frames_half/{video_name}'
    return len(os.listdir(video_dir))


def create_labels(label_name, label_fields):
    video_dir = "../input/video_features"
    save_dir = f'../input/{label_name}_labels'
    os.makedirs(save_dir, exist_ok=True)

    num_classes = len(label_fields)

    df = pd.read_csv('../input/folds_all.csv')
    for video, video_df in df.groupby('video_id'):
        video_len = get_video_frames(video)
        video_labels = np.zeros((video_len, num_classes))

        event_df = video_df[['frame', label_name]
                            ][~video_df['event'].isin(['start', 'end'])]
        event_df['label'] = event_df[label_name].map(
            lambda x: label_fields.index(x))

        for (idx, frame, _, label) in event_df.itertuples():
            video_labels[frame, label] = 1

        save_path = f'{save_dir}/{video}.npy'
        np.save(save_path, video_labels)
        
        # make time interval masks
        video_masks = np.zeros((video_len,))
        period_df = video_df[['frame', 'event']][video_df['event'].isin(['start', 'end'])]
        assert all(period_df['event'] != period_df['event'].shift(1))
        assert len(period_df) % 2 == 0
        start_end = period_df['frame'].values.reshape(-1, 2)
        for start, end in start_end:
            video_masks[start:end] = 1
        save_path = f'{save_dir}/{video}_mask.npy'
        np.save(save_path, video_masks)
        

if __name__ == "__main__":
    create_labels(
        label_name='event_attributes',
        label_fields=event_attribute_classes,)
    create_labels(
        label_name='event',
        label_fields=event_classes,
    )
