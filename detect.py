#############################################
# SMOKE DETECTOR V2.0 "SMOKY"
#
# Author: Daniel Eldan R.
# Date  : 12-2022
# Mail  : deldanr@gmail.com
# Name  : Detect
# Desc  : Load the model, run the inference, send telegram msg and post tweet.
############################################

import argparse
import time
from pathlib import Path
import datetime

import cv2
import torch
from numpy import random

#
# Load the YOLO base libraries
#
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from tinydb import TinyDB, Query
from tinydb.operations import increment

#
# IMPORT LOCAL FUNCTIONS TO SEND A TELEGRAM MSG AND POST A TWEET
#
import telegram_send
import tweet as tw
import os
from io import BytesIO
from PIL import Image

db = TinyDB("static/data.json")
Todo = Query()


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    detecciones = n
                db.update(increment('Detectadas'), Todo.Detectadas.exists())

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img or view_img:  # Add bbox to image
                        label = "Humo: " + f'{conf:.2f}' #f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=[0,0,255], line_thickness=3) #colors[int(cls)]

            # Print time (inference + NMS)
            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            db.update(increment('Procesadas'), Todo.Procesadas.exists())

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    #print(f" The image with the result is saved in: {save_path}")
            
            #if detecciones > 0:
            #    publish_result(save_path)
                

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")
        
    print(f'Done. ({time.time() - t0:.3f}s)')

def publish_result(path):
    
    x = datetime.datetime.now()

    print("********************INCENDIO DETECTADO... PUBLICANDO**********************")

      # Send then Telegram message
    telegram_send.send(messages=["INCENDIO FORESTAL DETECTADO"])
    string2 = str(x.day) + "-" + str(x.month) + "-" + str(x.year) + " | " + str(x.hour) + ":" + str(x.minute) + ":" + str(x.second)
    telegram_send.send(messages=[string2])

    with open(path, "rb") as f:
        telegram_send.send(images=[f])

        # Example image manipulation
    path2 = os.path.abspath(path)
    img = Image.open(path2)

        # Save image in-memory
    b = BytesIO()
    img.save(b, "PNG")
    b.seek(0)

        # Upload media to Twitter APIv1.1
    ret = tw.api.media_upload(filename="dummy_string", file=b)
    string_tw = "INCENDIO FORESTAL DETECTADO\n" + string2
        # Attach media to tweet
    tw.api.update_status(
        media_ids=[ret.media_id_string], status=string_tw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='model/Smoky5.0.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='static/test', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold') # 25% confidence
    parser.add_argument('--iou-thres', type=float, default=0.40, help='IOU threshold for NMS') # 40% IOU
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='static/test', help='save results to project/name')
    parser.add_argument('--name', default='', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
