from ultralytics import YOLO
#from ultralytics.utils.callbacks.custom import EpochCallback

# Load a model
#model = YOLO('runs/pose/pose35/weights/last.pt') # build a new model from YAML
model = YOLO('yolo26n-pose.yaml', task='pose')
# Train the model
results = model.train(data="/home/michael/data/data/board/ordered/board_ordered.yaml", 
                      epochs=50, 
                      imgsz=640, 
                      batch=8, 
                      task="pose", 
                      name="pose",
                      )#callbacks=[EpochCallback])
#results = model.train(data="pygo.yaml", epochs=100, imgsz=640, batch=6, task="pose", name="pose")