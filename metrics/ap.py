import numpy as np
import pandas as pd
from pandas.testing import assert_index_equal
from typing import Dict, Tuple

tolerances = {
    "challenge": [0.3, 0.4, 0.5, 0.6, 0.7],
    "play": [0.15, 0.20, 0.25, 0.30, 0.35],
    "throwin": [0.15, 0.20, 0.25, 0.30, 0.35],
}
# tolerances = {
#     "challenge": [1.0, 2.0, 3.0, 4.0, 5.0],
#     "play": [1.0, 2.0, 3.0, 4.0, 5.0],
#     "throwin": [1.0, 2.0, 3.0, 4.0, 5.0],
# }
            

def filter_detections(
        detections: pd.DataFrame, intervals: pd.DataFrame
) -> pd.DataFrame:
    """Drop detections not inside a scoring interval."""
    detection_time = detections.loc[:, 'time'].sort_values().to_numpy()
    intervals = intervals.to_numpy()
    is_scored = np.full_like(detection_time, False, dtype=bool)

    i, j = 0, 0
    while i < len(detection_time) and j < len(intervals):
        time = detection_time[i]
        int_ = intervals[j]

        # If the detection is prior in time to the interval, go to the next detection.
        if time < int_.left:
            i += 1
        # If the detection is inside the interval, keep it and go to the next detection.        
        elif time in int_:
            is_scored[i] = True
            i += 1
        # If the detection is later in time, go to the next interval.
        else:
            j += 1

    return detections.loc[is_scored].reset_index(drop=True)


def match_detections(
        tolerance: float, ground_truths: pd.DataFrame, detections: pd.DataFrame
) -> pd.DataFrame:
    """Match detections to ground truth events. Arguments are taken from a common event x tolerance x video evaluation group."""
    detections_sorted = detections.sort_values('score', ascending=False).dropna()

    is_matched = np.full_like(detections_sorted['event'], False, dtype=bool)
    gts_matched = set()
    for i, det in enumerate(detections_sorted.itertuples(index=False)):
        best_error = tolerance
        best_gt = None

        for gt in ground_truths.itertuples(index=False):
            error = abs(det.time - gt.time)
            if error < best_error and not gt in gts_matched:
                best_gt = gt
                best_error = error
            
        if best_gt is not None:
            is_matched[i] = True
            gts_matched.add(best_gt)

    detections_sorted['matched'] = is_matched

    return detections_sorted


def precision_recall_curve(
        matches: np.ndarray, scores: np.ndarray, p: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(matches) == 0:
        return [1], [0], []

    # Sort matches by decreasing confidence
    idxs = np.argsort(scores, kind='stable')[::-1]
    scores = scores[idxs]
    matches = matches[idxs]
    
    distinct_value_indices = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, matches.size - 1]
    thresholds = scores[threshold_idxs]
    
    # Matches become TPs and non-matches FPs as confidence threshold decreases
    tps = np.cumsum(matches)[threshold_idxs]
    fps = np.cumsum(~matches)[threshold_idxs]
    
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / p  # total number of ground truths might be different than total number of matches
    
    # Stop when full recall attained and reverse the outputs so recall is non-increasing.
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    # Final precision is 1 and final recall is 0
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def average_precision_score(matches: np.ndarray, scores: np.ndarray, p: int) -> float:
    precision, recall, _ = precision_recall_curve(matches, scores, p)
    # Compute step integral
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])

import time

from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
    
class MetricCalculator:
    def __init__(self, solution: pd.DataFrame, tolerances: Dict[str, float]):
        solution = solution[['video_id', 'time', 'event']]
        
        # Extract scoring intervals.
        self.intervals = (
            solution
            .query("event in ['start', 'end']")
            .assign(interval=lambda x: x.groupby(['video_id', 'event']).cumcount())
            .pivot(index='interval', columns=['video_id', 'event'], values='time')
            .stack('video_id')
            .swaplevel()
            .sort_index()
            .loc[:, ['start', 'end']]
            .apply(lambda x: pd.Interval(*x, closed='both'), axis=1)
        ) 
        # Extract ground-truth events.
        ground_truths = (
            solution
            .query("event not in ['start', 'end']")
            .reset_index(drop=True)
        )

        # Map each event class to its prevalence (needed for recall calculation)
        self.class_counts = ground_truths.value_counts('event').to_dict()

        # Create table of event-class x tolerance x video_id values
        self.aggregation_keys = pd.DataFrame(
            [(ev, tol, vid)
            for ev in tolerances.keys()
            for tol in tolerances[ev]
            for vid in ground_truths['video_id'].unique()],
            columns=['event', 'tolerance', 'video_id'],
        )
            
        self.ground_truths_grouped = (
            self.aggregation_keys
            .merge(ground_truths, on=['event', 'video_id'], how='left')
            .groupby(['event', 'tolerance', 'video_id'])
        )
        
        # Compute AP per event x tolerance group
        self.event_classes = ground_truths['event'].unique()
        
    def calc(self, submission: pd.DataFrame):
        submission = submission[['video_id', 'time', 'event', 'score']]
        submission = submission.sort_values(['video_id', 'time']).reset_index(drop=True)

        # Create table for detections with a column indicating a match to a ground-truth event
        detections = submission.assign(matched = False)

        with timer('scoring interval'):
            # Remove detections outside of scoring intervals
            detections_filtered = []
            for (det_group, dets), (int_group, ints) in zip(
                detections.groupby('video_id'), self.intervals.groupby('video_id')
            ):
                assert det_group == int_group
                detections_filtered.append(filter_detections(dets, ints))
            detections_filtered = pd.concat(detections_filtered, ignore_index=True)

            # Create match evaluation groups: event-class x tolerance x video_id
            detections_grouped = (
                self.aggregation_keys
                .merge(detections_filtered, on=['event', 'video_id'], how='left')
                .groupby(['event', 'tolerance', 'video_id'])
            )
        
        with timer('matching'):
            # Match detections to ground truth events by evaluation group
            detections_matched = []
            for key in self.aggregation_keys.itertuples(index=False):
                dets = detections_grouped.get_group(key)
                gts = self.ground_truths_grouped.get_group(key)
                detections_matched.append(
                    match_detections(dets['tolerance'].iloc[0], gts, dets)
                )
            detections_matched = pd.concat(detections_matched)
            
        ap_table = (
            detections_matched
            .query("event in @self.event_classes")
            .groupby(['event', 'tolerance']).apply(
            lambda group: average_precision_score(
            group['matched'].to_numpy(),
                    group['score'].to_numpy(),
                    self.class_counts[group['event'].iat[0]],
                )
            )
        )

        # Average over tolerances, then over event classes
        ap_per_events = {k: 0.0 for k in self.event_classes}
        ap_per_events.update(ap_table.groupby('event').mean().to_dict())
        mean_ap = np.mean(list(ap_per_events.values()))
        return mean_ap, ap_per_events
   
     
def event_detection_ap(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        tolerances: Dict[str, float],
) -> float:

    assert_index_equal(solution.columns, pd.Index(['video_id', 'time', 'event']))
    assert_index_equal(submission.columns, pd.Index(['video_id', 'time', 'event', 'score']))

    submission = submission.sort_values(['video_id', 'time']).reset_index(drop=True)
    
    # Extract scoring intervals.
    intervals = (
        solution
        .query("event in ['start', 'end']")
        .assign(interval=lambda x: x.groupby(['video_id', 'event']).cumcount())
        .pivot(index='interval', columns=['video_id', 'event'], values='time')
        .stack('video_id')
        .swaplevel()
        .sort_index()
        .loc[:, ['start', 'end']]
        .apply(lambda x: pd.Interval(*x, closed='both'), axis=1)
    )

    # Extract ground-truth events.
    ground_truths = (
        solution
        .query("event not in ['start', 'end']")
        .reset_index(drop=True)
    )

    # Map each event class to its prevalence (needed for recall calculation)
    class_counts = ground_truths.value_counts('event').to_dict()

    # Create table for detections with a column indicating a match to a ground-truth event
    detections = submission.assign(matched = False)

    # Remove detections outside of scoring intervals
    detections_filtered = []
    for (det_group, dets), (int_group, ints) in zip(
        detections.groupby('video_id'), intervals.groupby('video_id')
    ):
        assert det_group == int_group
        detections_filtered.append(filter_detections(dets, ints))
    detections_filtered = pd.concat(detections_filtered, ignore_index=True)

    # Create table of event-class x tolerance x video_id values
    aggregation_keys = pd.DataFrame(
        [(ev, tol, vid)
         for ev in tolerances.keys()
         for tol in tolerances[ev]
         for vid in ground_truths['video_id'].unique()],
        columns=['event', 'tolerance', 'video_id'],
    )

    # Create match evaluation groups: event-class x tolerance x video_id
    detections_grouped = (
        aggregation_keys
        .merge(detections_filtered, on=['event', 'video_id'], how='left')
        .groupby(['event', 'tolerance', 'video_id'])
    )
    ground_truths_grouped = (
        aggregation_keys
        .merge(ground_truths, on=['event', 'video_id'], how='left')
        .groupby(['event', 'tolerance', 'video_id'])
    )
    
    # Match detections to ground truth events by evaluation group
    detections_matched = []
    for key in aggregation_keys.itertuples(index=False):
        dets = detections_grouped.get_group(key)
        gts = ground_truths_grouped.get_group(key)
        detections_matched.append(
            match_detections(dets['tolerance'].iloc[0], gts, dets)
        )
    detections_matched = pd.concat(detections_matched)
    
    # Compute AP per event x tolerance group
    event_classes = ground_truths['event'].unique()
    ap_table = (
        detections_matched
        .query("event in @event_classes")
        .groupby(['event', 'tolerance']).apply(
        lambda group: average_precision_score(
        group['matched'].to_numpy(),
                group['score'].to_numpy(),
                class_counts[group['event'].iat[0]],
            )
        )
    )

    # Average over tolerances, then over event classes
    ap_per_events = {k: 0.0 for k in event_classes}
    ap_per_events.update(ap_table.groupby('event').mean().to_dict())
    mean_ap = np.mean(list(ap_per_events.values()))
    # mean_ap = ap_table.groupby('event').mean().mean()
    return mean_ap, ap_per_events


from numpy.testing import assert_almost_equal
def test_case1():
    solution = pd.DataFrame({
        "video_id": ["case1"] * 9, 
        "time": [1, 2, 3] + [11, 12, 13] + [21, 22, 23],
        "event": ["start", "play", "end"] + ["start", "challenge", "end"] + ["start", "throwin", "end"]
    })
    submission = pd.DataFrame({
        "video_id": ["case1"] * 3,
        "time": [2, 12, 22],
        "event": ["play", "challenge", "throwin"],
        "score": [1, 1, 1]
    })
    score, _ = event_detection_ap(solution=solution, submission=submission, tolerances=tolerances)
    assert_almost_equal(score, 1.0, decimal=5)
  
def test_case2():
    solution = pd.DataFrame({
        "video_id": ["case1"] * 9, 
        "time": [1, 2, 3] + [11, 12, 13] + [21, 22, 23],
        "event": ["start", "play", "end"] + ["start", "challenge", "end"] + ["start", "throwin", "end"]
    })
    submission = pd.DataFrame({
        "video_id": ["case1"] * 3,
        "time": [1.7, 12, 22],
        "event": ["play", "challenge", "throwin"],
        "score": [1, 1, 1]
    })
    score, _ = event_detection_ap(solution=solution, submission=submission, tolerances=tolerances)
    assert_almost_equal(score, 11/15, decimal=5)
          
def test_case3():
    solution = pd.DataFrame({
        "video_id": ["case1"] * 9, 
        "time": [1, 2, 3] + [11, 12, 13] + [21, 22, 23],
        "event": ["start", "play", "end"] + ["start", "challenge", "end"] + ["start", "throwin", "end"]
    })
    submission = pd.DataFrame({
        "video_id": ["case1"] * 8,
        "time": [2, 12, 22, 22.1, 22.2, 22.3, 22.4, 22.5],
        "event": ["play", "challenge", "throwin", "throwin", "throwin", "throwin", "throwin", "throwin"],
        "score": [1, 1, 1, 0.99, 0.99, 0.99, 0.99, 0.99]
    })
    score, score_per_events = event_detection_ap(solution=solution, submission=submission, tolerances=tolerances)
    print(score, score_per_events)
    assert_almost_equal(score, 1.0, decimal=5)
    
def test_case4():
    solution = pd.DataFrame({
        "video_id": ["case1"] * 9, 
        "time": [1, 2, 3] + [11, 12, 13] + [25, 26, 27],
        "event": ["start", "play", "end"] + ["start", "challenge", "end"] + ["start", "throwin", "end"]
    })
    submission = pd.DataFrame({
        "video_id": ["case1"] * 8,
        "time": [2, 12, 22, 22.1, 22.2, 22.3, 22.4, 22.5],
        "event": ["play", "challenge", "throwin", "throwin", "throwin", "throwin", "throwin", "throwin"],
        "score": [1, 1, 1, 0.99, 0.99, 0.99, 0.99, 0.99]
    })
    score, score_per_events = event_detection_ap(solution=solution, submission=submission, tolerances=tolerances)
    print(score, score_per_events)
    
def test_case5():
    solution = pd.DataFrame({
        "video_id": ["case1"] * 9, 
        "time": [1, 2, 3] + [11, 12, 13] + [21, 22.6, 26],
        "event": ["start", "play", "end"] + ["start", "challenge", "end"] + ["start", "throwin", "end"]
    })
    submission = pd.DataFrame({
        "video_id": ["case1"] * 8,
        "time": [2, 12, 22, 22.1, 22.2, 22.3, 22.4, 22.5],
        "event": ["play", "challenge", "throwin", "throwin", "throwin", "throwin", "throwin", "throwin"],
        "score": [1, 1, 1, 0.99, 0.99, 0.99, 0.99, 0.99]
    })
    score, score_per_events = event_detection_ap(solution=solution, submission=submission, tolerances=tolerances)
    print(score, score_per_events)
    
def test_case6():
    solution = pd.DataFrame({
        "video_id": ["case1"] * 9, 
        "time": [1, 2, 3] + [11, 12, 13] + [21, 22.1, 26],
        "event": ["start", "play", "end"] + ["start", "challenge", "end"] + ["start", "throwin", "end"]
    })
    submission = pd.DataFrame({
        "video_id": ["case1"] * 8,
        "time": [2, 12, 22, 22.1, 22.2, 22.3, 22.4, 22.5],
        "event": ["play", "challenge", "throwin", "throwin", "throwin", "throwin", "throwin", "throwin"],
        "score": [1, 1, 1, 0.99, 0.99, 0.99, 0.99, 0.99]
    })
    score, score_per_events = event_detection_ap(solution=solution, submission=submission, tolerances=tolerances)
    print(score, score_per_events)
    
def test_case7():
    solution = pd.DataFrame({
        "video_id": ["case1"] * 9, 
        "time": [1, 2, 3] + [11, 12, 13] + [21, 22, 26],
        "event": ["start", "play", "end"] + ["start", "challenge", "end"] + ["start", "throwin", "end"]
    })
    submission = pd.DataFrame({
        "video_id": ["case1"] * 8,
        "time": [2, 12, 22, 22.1, 22.2, 22.3, 22.4, 22.5],
        "event": ["play", "challenge", "throwin", "throwin", "throwin", "throwin", "throwin", "throwin"],
        "score": [1, 1, 1, 1, 1, 1, 1, 1]
    })
    score, score_per_events = event_detection_ap(solution=solution, submission=submission, tolerances=tolerances)
    print(score, score_per_events)
    
def test_case8():
    solution = pd.DataFrame({
        "video_id": ["case1"] * 9, 
        "time": [1, 2, 3] + [11, 12, 13] + [21, 22, 26],
        "event": ["start", "play", "end"] + ["start", "challenge", "end"] + ["start", "throwin", "end"]
    })
    submission = pd.DataFrame({
        "video_id": ["case1"] * 8,
        "time": [2, 12, 22, 22.1, 22.2, 22.3, 22.4, 22.5],
        "event": ["play", "challenge", "throwin", "throwin", "throwin", "throwin", "throwin", "throwin"],
        "score": [1, 1, 1, 0.99, 0.99, 0.99, 0.99, 0.99]
    })
    calculator = MetricCalculator(solution, tolerances)
    score, score_per_events = calculator.calc(submission)
    # score, score_per_events = event_detection_ap(solution=solution, submission=submission, tolerances=tolerances)
    print(score, score_per_events)

def test_case9():
    solution = pd.DataFrame({
        "video_id": ["case1"] * 9999, 
        "time": list(range(9999)),
        "event": ["start", "play", "end"] * 3333
    })
    submission = pd.DataFrame({
        "video_id": ["case1"] * 10000,
        "time": list(range(10000)),
        "event": ["play"] * 10000,
        "score": [1] * 10000
    })
    calculator = MetricCalculator(solution, tolerances)
    score, score_per_events = calculator.calc(submission)
    # score, score_per_events = event_detection_ap(solution=solution, submission=submission, tolerances=tolerances)
    print(score, score_per_events)
    
if __name__ == "__main__":
    # test_case3()
    # test_case4()
    # test_case5()
    # test_case6()
    # test_case7()
    test_case9()
    # import timeit
    # print(timeit.timeit(test_case1, number=10))
    # timeit.timeit(test_case2, number=1000)