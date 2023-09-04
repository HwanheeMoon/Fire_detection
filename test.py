import torchvision.transforms as transforms
import torch
from models.experimental import attempt_load
import cv2
from models.seqModel import SeqClassifier
from utils.general import non_max_suppression
from models.experimental import attempt_load
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import torch
import numpy as np
import cv2
from models.experimental import attempt_load
from utils.plots import Annotator, colors
from pathlib import Path


def nms(predictions, conf_thres=0.5, iou_thres=0.4):
    """
    Non-Maximum Suppression (NMS)를 적용하여 객체 탐지 결과 필터링.

    Args:
        predictions (Tensor): 객체 탐지 결과 텐서. 각 행은 [x1, y1, x2, y2, confidence, class] 형식.
        conf_thres (float): 객체 신뢰도의 최소 임계값.
        iou_thres (float): IoU (Intersection over Union)의 최소 임계값.

    Returns:
        filtered_predictions (Tensor): NMS를 적용한 객체 탐지 결과 텐서.
    """
    # 객체 신뢰도가 임계값 이하인 것은 제거
    print(predictions[:, 0:4])
    mask = predictions[:, 4] >= conf_thres
    predictions = predictions[1]

    if not predictions.size(0):
        return []

    # 객체들을 신뢰도 내림차순으로 정렬
    _, sorted_idx = predictions[:, 4].sort(0, descending=True)
    predictions = predictions[sorted_idx]

    keep = []  # 최종으로 남을 객체 인덱스

    while predictions.size(0):
        # 가장 신뢰도가 높은 객체 선택
        largest_box = predictions[0]

        # 선택된 객체와 나머지 모든 객체 간의 IoU 계산
        ious = bbox_iou(largest_box.unsqueeze(0), predictions[:, :4])

        # IoU가 임계값 이하인 객체들만 선택
        mask = ious <= iou_thres
        predictions = predictions[mask]

        # 선택된 객체는 결과에 추가
        keep.append(largest_box)

    if len(keep) > 0:
        return torch.stack(keep)
    else:
        return []

def bbox_iou(box1, box2):
    """
    두 개의 바운딩 박스 간의 IoU (Intersection over Union) 계산.

    Args:
        box1 (Tensor): 첫 번째 바운딩 박스 [x1, y1, x2, y2].
        box2 (Tensor): 두 번째 바운딩 박스 [x1, y1, x2, y2].

    Returns:
        iou (Tensor): IoU 값.
    """
    # 바운딩 박스 좌표 추출
    x1, y1, x2, y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    x3, y3, x4, y4 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 바운딩 박스 영역 계산
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    # 교차 영역 계산
    x_intersection = torch.max(torch.zeros_like(x1), torch.min(x2, x4) - torch.max(x1, x3))
    y_intersection = torch.max(torch.zeros_like(y1), torch.min(y2, y4) - torch.max(y1, y3))
    intersection = x_intersection * y_intersection

    # IoU 계산
    iou = intersection / (area1 + area2 - intersection)
    return iou



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

gru_hidden_size = 128

inc_classes = ['default', 'fire', 'fire increase', 'smoke', 'smoke increase']
classes = ['fire', 'smoke']

input_size = 512  # 이미지 특징의 크기
hidden_size = 128  # GRU hidden 크기
num_classes = len(classes)  # 분류할 클래스 수
num_layers = 2  # GRU 레이어 개수

model = SeqClassifier(input_size, hidden_size, len(inc_classes), num_layers, use_LSTM=True).to(device)
model.load_state_dict(torch.load('best_train-lstm.pt'))
#model.eval()

cap = cv2.VideoCapture('bus.mp4')

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

detect_model = attempt_load('model/best.pt')
detect_model.to(device).eval()


weights = 'model/best.pt'
data = 'model/fire_config.yaml'

conf_thres = 0.25
iou_thres = 0.45

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = preprocess(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        inc_pred = model(input_image).to(device)
        pred = detect_model(input_image)[0]


    _, pred2 = torch.max(inc_pred, 1)
    pred_index = pred2.item()



    if pred_index == 2 or pred_index == 4:
        cv2.putText(frame, f"{inc_classes[pred_index]}", (20, 70), cv2.FONT_HERSHEY_COMPLEX, 1.5,
                    (0, 0, 255), 2)
        top_left = (10, 10)  # (x, y) 좌표
        bottom_right = (frame.shape[1] - 10, frame.shape[0] - 10)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 10)

    if pred is not None and len(pred) > 0:
        pred = nms(pred)
        for det in pred:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = f"Class: {int(cls)}, Confidence: {conf:.2f}"
            color = (0, 255, 0)  # 초록색
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    cv2.imshow("TEST", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
