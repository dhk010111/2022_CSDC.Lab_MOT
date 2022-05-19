import numpy as np

""" 
단일 이미지에서 Bounding Box를 탐지
[Parameters 설명]
tlwh(top left의 좌표 & width height) : Bounding Box의 (x, y, w, h)
Confidence (Float) : Detector의 Confidence 점수
feature : feature 벡터는 이미지 안에 객체가 포함됨을 나타냄
"""




class Detection(object):
    def __init__(self, tlwh, confidence, class_name, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.class_name = class_name
        self.feature = np.asarray(feature, dtype=np.float32)

    def get_class(self):
        return self.class_name

    def to_tlbr(self):
        """
        Bounding Box의 형식을 (좌측 하단, 우측 상단)의 좌표로 변환 (min x, min y, max x, max y)
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """
        Bounding Box의 형식을 (중심, 가로세로 비율, 높이)로 변환 (center x, center y, ratio, height)
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
