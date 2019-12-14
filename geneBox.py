import os, pickle
import cv2
det_dir = "/mnt/data/Action/data/ucf24/ucf24/detections/CONV-rgb-01-060000/"
rgb_dir = "/mnt/data/Action/data/ucf24/ucf24/rgb-images"
action_class = ""
with open("/mnt/data/Action/data/ucf24/ucf24/cache/CONV-SSD-ucf24-rgb-bs-40-lr-00100/detection-060000.pkl", 'rb') as f:
    data = pickle.load(f)
    print(len(data[0]))

for root, _, file in os.walk(det_dir):
    pass
