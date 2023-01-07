import pandas as pd

FPS = 25.0
HEIGHT, WIDTH = 1080, 1920

video_fold_map = {
    '9a97dae4_1': 0,
    'ecf251d4_0': 0,
    '35bd9041_0': 1,
    '35bd9041_1': 1,
    '3c993bd2_0': 2,
    '3c993bd2_1': 2,
    '1606b0e6_0': 3,
    '1606b0e6_1': 3,
    '407c5a9e_1': 4,
    '4ffd5986_0': 4,
    'cfbe2e94_0': 5,
    'cfbe2e94_1': 5
}

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

if __name__ == "__main__":

    df = pd.read_csv('../input/dfl-bundesliga-data-shootout/train.csv')
    df['frame'] = df['time'] * FPS
    df['frame'] = df['frame'].round().astype(int)
    df['event_attributes'] = df['event_attributes'].fillna('""')
    df['event_attributes'] = df['event_attributes'].map(lambda x: "__".join(eval(x)))
    df['folds'] = df['video_id'].map(video_fold_map)
    df.to_csv("../input/folds_all.csv", index=False)

    start_df = df.query('event == "start"')
    end_df = df.query('event == "end"')

    rename_cols = ['time', 'frame']
    start_df = start_df.rename(columns={v: "start_" + v for v in rename_cols})
    end_df = end_df.rename(columns={v: "end_" + v for v in rename_cols})

    start_df = start_df.reset_index(drop=True)
    end_df = end_df.reset_index(drop=True)

    fold_df = pd.concat([start_df, end_df[['end_time', 'end_frame']]], axis=1)
    fold_df['num_frames'] = fold_df['end_frame'] - fold_df['start_frame']
    fold_df['duration'] = fold_df['end_time'] - fold_df['start_time']
    fold_df['num_frames'].max()
    fold_df.to_csv('../input/folds.csv', index=False)
