# CRC_MOT

YOLO v4-Deepsort를 이용한 Multi Object Tracking 구현







detection.py : Detection 기반 클래스

kalman_filter.py : 이미지 공간 필터링을 위한 칼만 필터

linear_assignment.py : 선형 matching 모듈(min cost, matching cascade)

iou_matching.py : IOU 행렬 모듈

nn_matching.py : Nearest neighbor matching 행렬 모듈

track.py : Track 클래스는 단일 이미지 트랙 데이터를 포함
