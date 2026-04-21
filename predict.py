import os
from ultralytics import YOLO


model = YOLO("yolo26n.pt")  # pretrained YOLO26n model

# Run batched inference on a list of images
images = [x for x in os.listdir('/home/michael/data/data/stone/pygo_stone/test/images/')]

results = model(images)

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk