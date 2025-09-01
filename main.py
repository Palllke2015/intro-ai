from ultralytics import YOLO

model = YOLO("yolo11l.pt")

train_results = model.train(
    data="data.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    # device='cpu'
)

metrics = model.val()   # evaluate
model.predict("samples/", conf=0.25, save=True)  # inference
model.export(format="onnx")  # export