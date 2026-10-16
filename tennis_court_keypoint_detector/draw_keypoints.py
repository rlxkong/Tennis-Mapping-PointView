import supervision as sv
import numpy as np
import cv2

class CourtKeypointsDrawer:
    def __init__(self):
        self.keypoint_color = (0, 0, 255) 
        self.label_color = (0, 0, 255) 
        self.radius = 4                     # Dot radius
        self.thickness = -1                 # Filled circle
        self.font_scale = 0.45
        self.font_thickness = 1
    
    def draw(self, frames, court_keypoints):
        output_frames = []

        for index, frame in enumerate(frames):
            annotate_frame = frame.copy()
            keypoints = court_keypoints[index]

            for i, (x, y) in enumerate(keypoints):
                x = int(x)
                y = int(y)

                # Draw a filled circle at (x, y)
                cv2.circle(annotate_frame, (x, y), self.radius, self.keypoint_color, self.thickness)

                # Draw the label (like "0", "1", ...) slightly offset
                label = str(i)
                cv2.putText(annotate_frame, label, (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                            self.label_color, self.font_thickness, cv2.LINE_AA)

            output_frames.append(annotate_frame)

        return output_frames