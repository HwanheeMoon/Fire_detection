import cv2
import os
import platform
import sys
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import torch
import numpy as np
import winsound as sd

def beepsound():
    fr = 1000    # range : 37 ~ 32767
    du = 50   # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)


def fire_prediction():
    weights = 'model/best.pt'
    data = 'model/fire_config.yaml'
    device = select_device('')

    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((640, 480), s=stride)

    webcam = '0'.isnumeric() or '0'.endswith('.streams')

    view_img = check_imshow(warn=True)
    dataset = LoadStreams('0', img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
    bs = len(dataset)

    conf_thres = 0.25
    iou_thres = 0.45


    #model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            annotator = Annotator(im0, line_width=2, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                print("Fire Detected!!!")
                beepsound()
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh)  # label format
                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = "Fire"
                        annotator.box_label(xyxy, None, color=colors(c, True))

            im0 = annotator.result()
            if view_img:
                cv2.imshow("Monitoring...", im0)
                k = cv2.waitKey(1)  # 1 millisecond

                if k == 27:
                    exit()


    cv2.destroyAllWindows()
