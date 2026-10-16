from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    #take the model we use
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_position(self, ball_positions):
        """predicts ball's path for when the ball isn't being detected"""
        #gets the list of bounding boxes and returns empty list if no detections
        ball_positions = [x.get(1, []) for x in ball_positions]
        #converts list to dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        #interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()   #handle not getting detections in first frame

        #convert to format before --> list of dict
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    #detects on multiple frames
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []
        
        #read stub and don't detect each frame
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections


        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        #if the stub_path exists we will save it
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    #tracks across frames
    def detect_frame(self, frame):
        #can detect frames before
        results = self.model.predict(frame, conf=0.2)[0]

        #key is ball id and output is bounding box
        ball_dict = {}
        for box in results.boxes:
            #xyxy = bounding box format (x min y min x max y max)
            result = box.xyxy.tolist()[0]

            #makes sure only ball bounding box are tracked
            ball_dict[1] = result

        return ball_dict
    
    def draw_bboxes(self, video_frames, ball_detections):
        """draws the bounding boxes on the balls"""
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            #Draw Bounding boxes
            for track_id, bbox in ball_dict.items():
                #get the xyxy and then draw the rectangle accordingly
                x1, y1, x2, y2 = bbox

                #labelling (id, min x, miny)
                cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2) 
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames