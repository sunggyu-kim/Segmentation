'''
모델을 수정하는 일반적인 상황
1. 사전 훈련된 모델에서 시작, 마지막 레이어만 미세 조정할때
2. 모델의 backbone(신경망 구조 - resnet18, transformer ...)을 다른것으로 교체하고 싶을때
'''

#Case 1 - 사전 훈련된 모델에서 시작
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

#Finetuning from a pretrained model

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2 #1 class (person) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

#replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)




#Case 2

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# load a pre-trained model for classification and return
# only the features
backbone = torchvision.models.mobilenet_v2(weights = 'DEFAULT').features
# ``FasterRCNN``은 backbone에서 output의 수를 알아야한다. mobilenet_v2의 경우 1280이다
backbone.out_channels = 1280

#let's make the RPN generate 5 X 3 anchors per spatial location,
#Faster R-CNN의 구조를 알아야 짤 수 있는 공간 
#RPN :Region Proposal Network 공간으로 특정 위치에 객체 유/무를 예측해줌
#Anchor : 이미지 위에 고정된 크기와 비율로 격자배치된 사각형 박스 : anchor를 기준으로 RPN이 객체 유/무를 판단해줌

anchor_generator = AnchorGenerator(
    sizes = ((32, 64, 128, 256, 512),),
    aspect_ratios = ((0.5, 1.0, 2.0),)
)

#MultiScaleRoIAlign : 여러개로 나눠진 ROI를 하나의 output featuremap에 맞게 크기를 변환시켜줌
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names = ['0'], #현재 사용하는 모델이 단 1개(mobilnet_v2)여서 1개만 featuremap으로 반환
    output_size = 7, 
    sampling_ratio = 2
)

# Faster-RCNN model에 조각 값들을 넣음
model = FasterRCNN(
    backbone,
    num_classes=2,
    rpn_anchor_generator = anchor_generator,
    box_roi_pool = roi_pooler
)

