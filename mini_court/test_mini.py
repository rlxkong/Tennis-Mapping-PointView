# test_minicourt.py
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__init__.py'))))  # Adjust if necessary to find the MiniCourt module
from mini_court import MiniCourt  # Assuming your file is named minicourt.py


# 1. Create a test frame
frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # Match your video resolution

# 2. Initialize MiniCourt
minicourt = MiniCourt(frame)

# 3. Define test keypoints (TL, TR, BL, BR of court in video)
# These should match actual court corners in your video
test_keypoints = [
    [200, 100],  # Top-left
    [1000, 100],  # Top-right
    [200, 600],   # Bottom-left
    [1000, 600]   # Bottom-right
]

# 4. Compute homography
H = minicourt.homography_matrix(test_keypoints)

# 5. Test the homography
if H is not None:
    print("Homography Matrix:")
    print(H)
    minicourt.test_homography(H, test_keypoints)
else:
    print("Failed to compute homography")

# 6. Test ball mapping
test_ball_positions = [
    [600, 350],  # Center of court (should map to center of mini-court)
    [200, 100],  # Top-left corner
    [1000, 600], # Bottom-right corner
    [800, 200]   # Somewhere in top-right
]

# Map and visualize
player_detections = [{}] * len(test_ball_positions)  # Empty player detections
_, ball_positions = minicourt.map_detections_to_mini_court(
    player_detections, test_ball_positions, test_keypoints)

print("\nBall Position Mapping Results:")
for orig, mapped in zip(test_ball_positions, ball_positions):
    print(f"Original: {orig} -> Mapped: {mapped}")

# Visualize
frames = [frame.copy() for _ in test_ball_positions]
result_frames = minicourt.draw_mini_court_with_detections(frames, player_detections, ball_positions)

for i, res_frame in enumerate(result_frames):
    cv2.putText(res_frame, f"Test {i+1}", (50,50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow(f"Test Case {i+1}", res_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()