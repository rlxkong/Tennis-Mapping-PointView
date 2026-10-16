from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    #take the model we use
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.track_history = {}  # {track_id: [(x, y), ...]}
        self.max_history = 10  # how many past positions to keep
        self.motion_threshold = 2  # tweak this based on your video

    def choose_and_filter_players(self, court_keypoints, player_detections):
        """finds the players closest to the court based on the keypoints and only detects them"""
        frame_to_choose = 15 if len(player_detections) > 15 else 0
        player_detections_first_frame = player_detections[frame_to_choose]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)

        if not chosen_player:
            print("[WARNING] No players chosen after filtering; returning all detections.")
            return player_detections

        filtered_player_detections = []     #implement doubles here
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections


    def choose_players(self, court_keypoints, player_dict):
        """chooses the players that will end up showing up"""
        print(f"[DEBUG] Current track_history keys: {list(self.track_history.keys())}")
        print(f"[DEBUG] player_dict keys: {list(player_dict.keys())}")

        distances = []  #btwn each player and court
        for track_id, bbox in player_dict.items():

            # Motion filter
            if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
                continue  # Not enough data

            # Compute total motion
            positions = self.track_history[track_id]
            motion = sum(
                measure_distance(positions[i], positions[i - 1])
                for i in range(1, len(positions))
            )

            print(f"[DEBUG] Track {track_id} motion: {motion:.2f}")

            if motion < self.motion_threshold:
                continue  # Skip static people like line judges

            player_center = get_center_of_bbox(bbox)

            #calculate distance from player and court
            min_distance = float('inf')
            flat_keypoints = [coord for point in court_keypoints for coord in point]
            for i in range(0, len(flat_keypoints), 2):
                court_keypoint = (flat_keypoints[i], flat_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            #list of tuples with track_id and min_distances
            distances.append((track_id, min_distance))


        # Not enough candidates left? Fallback or return fewer
        if len(distances) < 2:
            return [d[0] for d in distances]  # can return 0, 1, or 2
        
        #sort the distances in ascending order
        distances.sort(key = lambda x: x[1])
        #choose the first 2 tracks --> change for doubles
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players


    #detects on multiple frames
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []
        
        #read stub and don't detect each frame
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)

            for track_id, bbox in player_dict.items():
                center = get_center_of_bbox(bbox)
                
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                self.track_history[track_id].append(center)
                print(f"[DEBUG] Track {track_id} history length: {len(self.track_history[track_id])}")
                
                # Keep only the last N positions
                if len(self.track_history[track_id]) > self.max_history:
                    self.track_history[track_id] = self.track_history[track_id][-self.max_history:]

            player_detections.append(player_dict)

        #if the stub_path exists we will save it
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    # #tracks across frames
    # def detect_frame(self, frame):
    #     #can detect frames before
    #     results = self.model.track(frame, persist=True)[0]
    #     #map id to names
    #     id_name_dict = results.names

    #     #key is player id and output is bounding box
    #     player_dict = {}
    #     for box in results.boxes:
    #         track_id = int(box.id.tolist()[0])
    #         #xyxy = bounding box format (x min y min x max y max)
    #         result = box.xyxy.tolist()[0]
    #         object_cls_id = box.cls.tolist()[0]
    #         object_cls_name = id_name_dict[object_cls_id]
    #         #makes sure only people bounding box are tracked
    #         if object_cls_name == "person":
    #             player_dict[track_id] = result

    #     return player_dict

###########

    #tracks across frames
    def detect_frame(self, frame):
        #can detect frames before
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        #key is player id and output is bounding box
        player_dict = {}
        for box in results.boxes:
            print(f"[DEBUG] box.id: {box.id}, box.cls: {box.cls}")
            if box.id is None:
                continue

            track_id = int(box.id.tolist()[0])
            class_id = int(box.cls.tolist()[0])

            # Skip if class_id is 0 (background or unwanted class)
            if class_id == 0:
                continue  # Skip this detection
            
            #xyxy = bounding box format (x min y min x max y max)
            result = box.xyxy.tolist()[0]
            
            # Assign to dictionary
            player_dict[track_id] = result
        
        return player_dict
    
    def draw_bboxes(self, video_frames, player_detections):
        """draws the bounding boxes on the players"""
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            #Draw Bounding boxes
            for track_id, bbox in player_dict.items():
                #get the xyxy and then draw the rectangle accordingly
                x1, y1, x2, y2 = bbox

                #labelling (id, min x, miny)
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) 
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            output_video_frames.append(frame)

        return output_video_frames
    
    def find_serving_player(self, player_detections, court_keypoints):
        """Finds the serving player based on their position relative to the court keypoints."""
        if not player_detections:
            return None
        
        # Get the first frame's detections
        first_frame_detections = player_detections[0]
        
        # Calculate the center of the court keypoints
        court_center_x = (court_keypoints[0][0] + court_keypoints[1][0]) / 2
        court_center_y = (court_keypoints[0][1] + court_keypoints[1][1]) / 2
        
        # Find the player closest to the court center
        closest_player = None
        min_distance = float('inf')
        
        for track_id, bbox in first_frame_detections.items():
            player_center = get_center_of_bbox(bbox)
            distance = measure_distance(player_center, (court_center_x, court_center_y))
            
            if distance < min_distance:
                min_distance = distance
                closest_player = track_id
        
        return closest_player