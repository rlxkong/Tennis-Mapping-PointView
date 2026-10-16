import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import(convert_pixel_distance_to_meters, 
                  convert_meters_to_pixel_distance, 
                  get_foot_position_from_bbox, 
                  get_closest_keypoint_index, 
                  get_center_of_bbox, measure_distance)

class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_bg_box_position(frame)
        self.set_mini_court_positions()
        self.set_mini_court_ball_positions()
        self.set_court_drawing_keypoints()
        self.set_court_lines()

        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.previous_ball_pos = None      #For smoothing

    def convert_meters_to_pixel_distance(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_widthp)

    def set_court_drawing_keypoints(self):
        drawing_keypoints = [0]*28

        #point 0 (top left)
        drawing_keypoints[0], drawing_keypoints[1] = int(self.court_start_xp), int(self.court_start_yp)
        # point 1 (top-right)
        drawing_keypoints[2] = int(self.court_start_xp + self.court_drawing_widthp)
        drawing_keypoints[3] = int(self.court_start_yp)
        #point 2 (bottom left)
        drawing_keypoints[4] = int(self.court_start_xp)
        drawing_keypoints[5] = self.court_start_yp + self.convert_meters_to_pixel_distance(constants.HALF_COURT*2)
        # point 3
        drawing_keypoints[6] = drawing_keypoints[0] + self.court_drawing_widthp
        drawing_keypoints[7] = drawing_keypoints[5] 
        # #point 4
        drawing_keypoints[8] = drawing_keypoints[0] +  self.convert_meters_to_pixel_distance(constants.DOUBLE_ALLEY)
        drawing_keypoints[9] = drawing_keypoints[1] 
        # #point 5
        drawing_keypoints[10] = drawing_keypoints[4] + self.convert_meters_to_pixel_distance(constants.DOUBLE_ALLEY)
        drawing_keypoints[11] = drawing_keypoints[5] 
        # #point 6
        drawing_keypoints[12] = drawing_keypoints[2] - self.convert_meters_to_pixel_distance(constants.DOUBLE_ALLEY)
        drawing_keypoints[13] = drawing_keypoints[3] 
        # #point 7
        drawing_keypoints[14] = drawing_keypoints[6] - self.convert_meters_to_pixel_distance(constants.DOUBLE_ALLEY)
        drawing_keypoints[15] = drawing_keypoints[7] 
        # #point 8
        drawing_keypoints[16] = drawing_keypoints[8] 
        drawing_keypoints[17] = drawing_keypoints[9] + self.convert_meters_to_pixel_distance(constants.NO_MANS_LAND)
        # # #point 9
        drawing_keypoints[18] = drawing_keypoints[16] + self.convert_meters_to_pixel_distance(constants.SINGLE_LINE_WIDTH)
        drawing_keypoints[19] = drawing_keypoints[17] 
        # #point 10
        drawing_keypoints[20] = drawing_keypoints[10] 
        drawing_keypoints[21] = drawing_keypoints[11] - self.convert_meters_to_pixel_distance(constants.NO_MANS_LAND)
        # # #point 11
        drawing_keypoints[22] = drawing_keypoints[20] +  self.convert_meters_to_pixel_distance(constants.SINGLE_LINE_WIDTH)
        drawing_keypoints[23] = drawing_keypoints[21] 
        # # #point 12
        drawing_keypoints[24] = int((drawing_keypoints[16] + drawing_keypoints[18])/2)
        drawing_keypoints[25] = drawing_keypoints[17] 
        # # #point 13
        drawing_keypoints[26] = int((drawing_keypoints[20] + drawing_keypoints[22])/2)
        drawing_keypoints[27] = drawing_keypoints[21] 

        self.drawing_keypoints=drawing_keypoints

    def set_court_lines(self):
        self.lines = [
            #vertical lines
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            (12,13),

            #horizontal lines
            (0,1),
            (8,9),
            (10,11),
            (2,3)
        ]    

    def set_mini_court_positions(self):
        """ the actual court in the minimap"""

        self.court_start_xp = self.start_x + self.padding_court
        self.court_start_yp = self.start_y + self.padding_court
        self.court_end_xp = self.end_x - self.padding_court
        self.court_end_yp = self.end_y - self.padding_court
        self.court_drawing_widthp = self.court_end_xp - self.court_start_xp

    def set_mini_court_ball_positions(self):
        
        # GET THIS OUT FOR BALL ACCCURATE
        self.court_start_xb = self.padding_court  # e.g., 20
        self.court_start_yb = self.padding_court  # e.g., 20
        self.court_end_xb = self.drawing_rectangle_width - self.padding_court  # e.g., 230
        self.court_end_yb = self.drawing_rectangle_height - self.padding_court  # e.g., 480
        self.court_drawing_widthb = self.court_end_xb - self.court_start_xb

    def set_canvas_bg_box_position(self, frame):
        """ the background underlying where the minicourt is"""
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self, frame):
        #draw lines
        for line in self.lines:
            start_point = (int(self.drawing_keypoints[line[0]*2]), int(self.drawing_keypoints[line[0]*2+1]))
            end_point = (int(self.drawing_keypoints[line[1]*2]), int(self.drawing_keypoints[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0,0,0), 2)

        #draws keypoints
        for i in range(0, len(self.drawing_keypoints), 2):
            x = int(self.drawing_keypoints[i])
            y = int(self.drawing_keypoints[i+1])
            cv2.circle(frame, (x,y), 5, (0,0,255), -1)

        #draw net
        net_start_point = (self.drawing_keypoints[0], int((self.drawing_keypoints[1]+self.drawing_keypoints[5])/2))
        net_end_point = (self.drawing_keypoints[2], int((self.drawing_keypoints[1]+self.drawing_keypoints[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame
    
    
    def draw_background_rectangle(self, frame):
        """
        Creates a semi-transparent white rectangle that won't blend with content drawn later
        """
        # Create the result frame
        result = frame.copy()
        
        # Create overlay for just the rectangle area
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        
        # Blend only the rectangle region
        alpha = 0.5
        roi = result[self.start_y:self.end_y, self.start_x:self.end_x]
        overlay_roi = overlay[self.start_y:self.end_y, self.start_x:self.end_x]
        
        # Apply weighted addition only to the region of interest
        result[self.start_y:self.end_y, self.start_x:self.end_x] = cv2.addWeighted(roi, 1-alpha, overlay_roi, alpha, 0)
        
        return result

    def draw_mini_court(self, frames):
        output_frames =[]
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames
    
    def get_start_point_of_mini_court(self):
        return(self.court_start_xp, self.court_start_yp)
    def get_width_of_mini_court(self):
        return self.court_drawing_widthp
    def get_court_drawing_keypoints(self):
        return self.drawing_keypoints
    
    def get_frames_near_bounces(self, ball_detections, bounce_frames):
        """
        Returns ball positions where:
        - Known detections are kept
        - Missing detections between bounce points are linearly interpolated
        """
        frames = [None] * len(ball_detections)
        bounce_frames = sorted(list(bounce_frames))
        
        # MAKE IT SO THE BALL STARTS AT THE PLAYER POSTION WHERE THE PERSON
        # CLOSER TO THE MIDDLE IS THE ONE WHO SERVES (AND THEREFORE WHERE THE BALL IS)

        for i in range(len(bounce_frames) - 1):
            start_idx = bounce_frames[i]
            end_idx = bounce_frames[i + 1]
            
            # Get actual positions
            start_pos = ball_detections[start_idx]
            end_pos = ball_detections[end_idx]
            
            if start_pos is None or end_pos is None:
                continue  # Skip if data is missing
            
            x1, y1 = start_pos
            x2, y2 = end_pos
            total = end_idx - start_idx
            
            # Interpolate linearly for each frame between the bounces
            for j in range(start_idx, end_idx + 1):
                t = (j - start_idx) / total
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                frames[j] = (x, y)
        
        # Optionally: fill in the first segment before the first bounce
        if bounce_frames:
            first_idx = bounce_frames[0]
            for j in range(0, first_idx):
                frames[j] = ball_detections[j]

        # Optionally: fill in after last bounce
        if bounce_frames and bounce_frames[-1] < len(ball_detections):
            last_idx = bounce_frames[-1]
            for j in range(last_idx, len(ball_detections)):
                frames[j] = ball_detections[j]
        
        return frames
    
    def get_frames_with_hits(self, ball_detections, bounce_frames, hit_frames):
        """
        Returns ball positions where:
        - Known detections are kept
        - Missing detections between points are linearly interpolated
        - Hit frames change the interpolation direction
        """
        frames = [None] * len(ball_detections)
        
        # Combine and sort all key frames (bounces and hits)
        key_frames = sorted(list(set(bounce_frames).union(set(hit_frames))))
        
        # First fill in all known detections
        for i, pos in enumerate(ball_detections):
            if pos is not None:
                frames[i] = pos
        
        for i in range(len(key_frames) - 1):
            start_idx = key_frames[i]
            end_idx = key_frames[i + 1]
            
            # Get actual positions (skip if missing)
            if frames[start_idx] is None or frames[end_idx] is None:
                continue
                
            x1, y1 = frames[start_idx]
            x2, y2 = frames[end_idx]
            total = end_idx - start_idx
            
            # Only interpolate if we have valid positions
            if None not in (x1, y1, x2, y2) and total > 0:
                for j in range(start_idx + 1, end_idx):
                    if frames[j] is None:  # Only fill missing frames
                        t = (j - start_idx) / total
                        x = x1 + t * (x2 - x1)
                        y = y1 + t * (y2 - y1)
                        frames[j] = (x, y)
        
        return frames

    def homography_matrix(self, keypoints):
        """
        Computes the homography matrix to map points from the video court to the mini court.

        :param keypoints: List of four (x, y) coordinates representing the corners of the court in the video frame.
        :return: Homography matrix (3x3 numpy array) or None if computation fails.
        """
        list_of_keypoints = [list(coord) for coord in keypoints]

        video_pts = np.array(list_of_keypoints[:4], dtype=np.float32)

        # Mini-court corners (TL, TR, BL, BR)
        court_pts = np.array([
            [self.court_start_xb, self.court_start_yb],  # TL
            [self.court_end_xb, self.court_start_yb],    # TR
            [self.court_start_xb, self.court_end_yb],    # BL
            [self.court_end_xb, self.court_end_yb]       # BR
        ], dtype=np.float32)

        # Explicitly select the four corners: top-left, top-right, bottom-left, bottom-right
        minicourt_corners = [
            [int(self.drawing_keypoints[0]), int(self.drawing_keypoints[1])],   # top-left
            [int(self.drawing_keypoints[2]), int(self.drawing_keypoints[3])],   # top-right
            [int(self.drawing_keypoints[4]), int(self.drawing_keypoints[5])],   # bottom-left
            [int(self.drawing_keypoints[6]), int(self.drawing_keypoints[7])]    # bottom-right
        ]
        model_court_pts = np.array(
            minicourt_corners, 
            dtype=np.float32)
        
        H, status = cv2.findHomography(video_pts, model_court_pts, cv2.RANSAC, 5.0)

        #GET THIS OUT FOR BALL ACCURATE
        # Normalize coordinates to [0,1] range
        video_norm = video_pts / np.array([self.frame_width, self.frame_height])  # Assuming 1280x720 video
        court_norm = (court_pts - np.array([self.start_x, self.start_y])) / \
                    np.array([self.drawing_rectangle_width, self.drawing_rectangle_height])

        # Compute normalized homography
        H2_norm, _ = cv2.findHomography(video_norm, court_norm)
        
        # Convert back to pixel coordinates
        scale_matrix = np.array([
            [self.drawing_rectangle_width, 0, self.start_x],
            [0, self.drawing_rectangle_height, self.start_y],
            [0, 0, 1]
        ])
        
        H2 = scale_matrix @ H2_norm @ np.linalg.inv(np.array([
            [1280, 0, 0],
            [0, 720, 0],
            [0, 0, 1]
        ]))
        
        return H, H2

    def map_detections_to_mini_court(self, player_detections, ball_detections, keypoints):
        """
        Maps player and ball detections to the mini court using homography.
        :param player_detections: List of dictionaries containing player detections per frame
                                [{player_id: bbox}, ...]
        :param ball_detections: List of ball positions per frame
        :param keypoints: Keypoints used for homography transformation
        :return: Tuple of (player_positions, ball_positions) where:
                - player_positions is a list of dictionaries in mini court coordinates
                - ball_positions is a list of positions in mini court coordinates
        """
        H, H2 = self.homography_matrix(keypoints)
        if H is None:
            return [{} for _ in player_detections]
        if H2 is None:
            return [None for _ in ball_detections]
        
        # Map player detections for each frame
        player_positions = []
        for frame_detections in player_detections:
            frame_positions = {}
            if isinstance(frame_detections, dict):  # Handle frame with detections
                for player_id, bbox in frame_detections.items():
                    try:
                        x, y = get_foot_position_from_bbox(bbox)
                        point = np.array([[x, y]], dtype=np.float32)
                        transformed_point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), H).reshape(2)
                        frame_positions[player_id] = transformed_point.tolist()
                    except Exception as e:
                        print(f"Error mapping player {player_id}: {str(e)}")
                        frame_positions[player_id] = None
            player_positions.append(frame_positions)
        
        # Map ball detections for each frame
        # Debugging output
        print("Video court points:", keypoints[:4])
        print("Mini-court points:", [[self.drawing_keypoints[0], self.drawing_keypoints[1]],
                                [self.drawing_keypoints[2], self.drawing_keypoints[3]],
                                [self.drawing_keypoints[4], self.drawing_keypoints[5]],
                                [self.drawing_keypoints[6], self.drawing_keypoints[7]]])

        ball_positions = []
        for ball_pos in ball_detections:
            if ball_pos is not None and len(ball_pos) >= 2:
                try:
                    # Ensure proper array shape for perspectiveTransform
                    point = np.array([[ball_pos[0], ball_pos[1]]], dtype=np.float32)
                    transformed = cv2.perspectiveTransform(point.reshape(1,1,2), H2)[0][0]
                    
                    # Add compensation if needed (adjust these values)
                    x = transformed[0] + self.start_x + 0  # Add offsets if necessary
                    y = transformed[1] + self.start_y + 0
                    
                    # # Apply smoothing if valid previous position exists
                    # if self.previous_ball_pos is not None:
                    #     x = 0.5 * x + 0.5 * self.previous_ball_pos[0]
                    #     y = 0.5 * y + 0.5 * self.previous_ball_pos[1]
                    
                    # CLAMP XY POSITOONT!!!!!! --> stays within the background rectangle
                    y = max(self.start_y, min(y, self.end_y))
                    x = max(self.start_x, min(x, self.end_x))

                    # Store raw transformed position (no clamping)
                    ball_positions.append([int(x), int(y)])
                    self.previous_ball_pos = (x, y)
                    
                except Exception as e:
                    print(f"Ball mapping error: {e}")
                    ball_positions.append(None)
                    self.previous_ball_pos = None
            else:
                ball_positions.append(None)
                self.previous_ball_pos = None
        
        return player_positions, ball_positions
    
    def draw_mini_court_with_detections(self, frames, player_positions, ball_positions):
        """
        Draws the mini court with player and ball detections - proper layering
        """
        output_frames = []
        
        for frame_idx, frame in enumerate(frames):
            # Start with the original frame and add semi-transparent background
            result = self.draw_background_rectangle(frame.copy())
            
            # Draw court lines on the background
            result = self.draw_court(result)
            
            # Get detections for current frame
            frame_players = player_positions[frame_idx] if frame_idx < len(player_positions) else {}
            frame_ball = ball_positions[frame_idx] if frame_idx < len(ball_positions) else None
            
            # Draw players on top of everything
            for player_id, pos in frame_players.items():
                if pos is not None and len(pos) >= 2:
                    try:
                        x, y = int(pos[0]), int(pos[1])
                        x = max(self.start_x, min(x, self.end_x))
                        y = max(self.start_y, min(y, self.end_y))
                        
                        color = (0, 255, 0) if player_id == 1 else (0, 255, 0)
                        cv2.circle(result, (x, y), 8, (0,0,0), -1)
                        cv2.circle(result, (x, y), 6, color, -1)
                        cv2.putText(result, f"P{player_id}", (x+10, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except Exception as e:
                        print(f"Error drawing player {player_id}: {str(e)}")
            
            # Draw ball on top of everything
            if frame_ball is not None and len(frame_ball) >= 2:
                try:
                    x, y = int(frame_ball[0]), int(frame_ball[1])
                    cv2.circle(result, (x, y), 6, (0, 0, 0), -1)
                    cv2.circle(result, (x, y), 4, (0, 255, 255), -1)
                except Exception as e:
                    print(f"Error drawing ball: {str(e)}")
            
            # Add debug information
            # if frame_idx < len(ball_positions):
            #     ball_pos = ball_positions[frame_idx]
            #     debug_text = f"Ball: {ball_pos if ball_pos else 'None'}"
            #     cv2.putText(result, debug_text, (self.start_x, self.start_y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            
            output_frames.append(result)
        
        return output_frames
    
    #FOR DEBUGGING EHHEHFHEFHEHFEHF
    ## DIDNT USE 
    def test_homography(self, H, test_points):
        """Test homography with known points and visualize results"""
        if H is None:
            print("No valid homography matrix")
            return
        
        # Create test debug frame
        debug_frame = np.zeros((self.end_y-self.start_y, self.end_x-self.start_x, 3), dtype=np.uint8)
        debug_frame[:] = (255, 255, 255)  # White background
        
        # Draw mini-court on debug frame
        self.draw_court(debug_frame)
        
        # Test points (TL, TR, BL, BR, Center)
        test_points_video = np.array([
            [[self.start_x, self.start_y]],    # Top-left
            [[self.end_x, self.start_y]],      # Top-right
            [[self.start_x, self.end_y]],      # Bottom-left
            [[self.end_x, self.end_y]],        # Bottom-right
            [[(self.start_x+self.end_x)//2, (self.start_y+self.end_y)//2]]  # Center
        ], dtype=np.float32)
        
        # Transform points
        transformed_points = cv2.perspectiveTransform(test_points_video, H)
        
        # Draw original and transformed points
        for i, (orig, trans) in enumerate(zip(test_points_video, transformed_points)):
            orig_x, orig_y = int(orig[0][0]), int(orig[0][1])
            trans_x, trans_y = int(trans[0][0]), int(trans[0][1])
            
            print(f"Point {i}: Video ({orig_x},{orig_y}) -> Mini-court ({trans_x},{trans_y})")
            
            # Draw original (red)
            cv2.circle(debug_frame, (orig_x-self.start_x, orig_y-self.start_y), 5, (0,0,255), -1)
            # Draw transformed (green)
            cv2.circle(debug_frame, (trans_x, trans_y), 5, (0,255,0), -1)
            # Draw line between them (blue)
            cv2.line(debug_frame, 
                    (orig_x-self.start_x, orig_y-self.start_y),
                    (trans_x, trans_y), 
                    (255,0,0), 1)
        
        cv2.imshow("Homography Test", debug_frame)
        cv2.waitKey(0)
        