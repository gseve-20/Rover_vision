from contextlib import nullcontext
from operator import truediv
import os
from pickle import TRUE
import cv2
import time as tim
import argparse

import torch
import model.detector
import utils.utils
import keyboard



def makeModel():
    #指定训练配置文件

    #print("Called")

    parser = argparse.ArgumentParser()
    """parser.add_argument('--data', type=str, default='', 
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='', 
                        help='The path of the .pth model to be transformed')
    parser.add_argument('--img', type=str, default='', 
                        help='The path of test image')"""

    opt = parser.parse_args()

    opt.data = 'data/coco.data'
    #opt.weights =  'weights/coco-290-epoch-0.780891ap-model.pth'
    #coco-200-epoch-0.611445ap-model.pth
    opt.weights =  'weights/coco-280-epoch-0.854864ap-model.pth'

    cfg = utils.utils.load_datafile(opt.data)
    #assert os.path.exists(opt.weights), "请指定正确的模型路径"
    #assert os.path.exists(opt.img), "请指定正确的测试图像路径"

    #模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    mod.load_state_dict(torch.load(opt.weights, map_location=device))

    #sets the module in eval node
    mod.eval()

    return mod, cfg, device
    
def print_test(frame, mod, cfg, device):
    #数据预处理
    ori_img = frame
    res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation = cv2.INTER_LINEAR) 
    img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
    img = torch.from_numpy(img.transpose(0,3, 1, 2))
    img = img.to(device).float() / 255.0

    #模型推理
    start = tim.perf_counter()
    preds = mod(img)
    end = tim.perf_counter()
    time = (end - start) * 1000.
    #print("forward time:%fms"%time)

    #特征图后处理
    output = utils.utils.handel_preds(preds, cfg, device)
    output_boxes = utils.utils.non_max_suppression(output, conf_thres = 0.1, iou_thres = 0.1)

    #加载label names
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
	    for line in f.readlines():
	        LABEL_NAMES.append(line.strip())
    
    h, w, _ = ori_img.shape
    scale_h, scale_w = h / cfg["height"], w / cfg["width"]

    #绘制预测框
    for box in output_boxes[0]:
        box = box.tolist()
       
        obj_score = box[4]
        if (int(box[5]) == 15 and float(box[4]) > 0.85):
            category = 'rock'
        elif(int(box[5]) == 16 and float(box[4]) >= 0.5):
            category = 'sample'
        else:
            continue
        #category = LABEL_NAMES[int(box[5])]

        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

        color = (216,53,149)
        if category == 'sample':
            color = (0,255,0)

        cv2.rectangle(ori_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, color, 2)	
        cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, color, 2)

    return ori_img


# we create the video capture object cap

mod, cfg, device = makeModel()
prev_frame_time = new_frame_time = 0
cap = cv2.VideoCapture(1)
isOn = False

if not cap.isOpened():
    raise IOError("We cannot open webcam")

while True:

    if keyboard.is_pressed("F"):
        isOn = False
    if keyboard.is_pressed("T"):
        isOn = True

    ret, frame = cap.read()
    # resize our captured frame if we need
    frame = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
        
    r_image = frame
    #r_image = cv2.flip(r_image,0)
    #r_image = cv2.flip(r_image,1)
    if (isOn == True):
        r_image = print_test(r_image, mod, cfg, device)


    new_frame_time = tim.time() 
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    fps = "FPS: " + str(int(fps))
    r_image = cv2.putText(r_image, fps, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    r_image = cv2.resize(r_image, (1500, 800))


    cv2.imshow("Web cam input", r_image)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()
    