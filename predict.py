import os
from ultralytics import YOLO
import cv2
import numpy as np
import math

run = 'pose'
weights_file = f"/home/michael/data/dev/ultralytics/runs/pose/{run}/weights/last.pt"
model_path = os.path.split(os.path.split(weights_file)[0])[0]
os.makedirs(os.path.join(model_path, 'predict'), exist_ok=True)
model = YOLO(weights_file)  # pretrained YOLO26n model

# Run batched inference on a list of images
image_folder = '/home/michael/data/data/stone/pygo_stone/test/images/'
images = [os.path.join(image_folder, x) for x in os.listdir(image_folder)]
images = [cv2.imread(x) for x in images]


for idx, image in enumerate(images):
    h,w,c = image.shape
    src = np.array([[0,0],[0,h], [w,0], [w,h]])
    rnd = np.random.randint(-250, 250, 8).reshape(4,2)
    rnd[np.random.choice(4,2)] = 0
    dst = src + rnd
    H,_ = cv2.findHomography(src,dst)
    image = cv2.warpPerspective(image, H, (w,h))
    factor = math.ceil(max(w/640, h/640))
    image = cv2.copyMakeBorder(image, 
                               (640*factor - h)//2, 
                               (640*factor - h)//2, 
                               (640*factor - w)//2, 
                               (640*factor - w)//2, 
                                cv2.BORDER_CONSTANT)
    results = model(image)
    result = results[0]
    # Process results list
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen
    result.save(filename=f"{model_path}/predict/image_{idx}.jpg")  # save to disk