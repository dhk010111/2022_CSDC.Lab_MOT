# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import linear_assignment


def iou(bbox, candidates):
    """
    IOU를 계산하는 함수
    [매개변수]
    Bounding Box(배열) : (좌측 상단, 너비 ,높이)의 좌표 (top left x, top left y, width, height)
    Candidates(배열) : Bounding Box의 row 하나당 하나씩 선별~

    [반환값]
    Bounding Box와 Candidates에 대한 Union 값
    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """
    IOU의 distance 행렬

    [매개변수]
    tracks(리스트) : 트랙의 리스트
    detections(리스트) : detections의 리스트
    track_indices(리스트) : 일치해야하는 트랙에 대한 리스트
    detection_indices(리스트) : 일치해야하는 detection에 대한 리스트

    [반환값]
    COST에 대한 행렬
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix
