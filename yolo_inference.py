from ultralytics import YOLO

model = YOLO("models/yolov5_last.pt")

# runs yolo on an image/video --> add save=True to save into the folder
result = model.predict("input_videos/pro3.mp4", conf=0.2, save=True)
# prints the different classes, boxes and stuff
# print(result)
# print("boxes: ")
# for box in result[0].boxes:
#     print(box)