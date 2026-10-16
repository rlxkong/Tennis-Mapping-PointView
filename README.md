# Tennis Mapping - PointView ESP

This repository is a Python tennis-video analysis project. Its current pipeline detects players, tracks the ball, maps court positions onto a mini-court view, estimates bounces and hits, and writes an annotated output video.

## Pipeline

The current entrypoint is main.py. It combines these project modules:

- trackers/ for player detection and tracking.
- TrackNet/ for tennis-ball tracking.
- tennis_court_keypoint_detector/ for court keypoint annotation and drawing.
- mini_court/ for mapping detected positions into a mini-court view.
- bounce_detector/ for bounce and hit estimates.
- utils/ for video and geometry helpers.

This repository contains the project source. It does not include match videos, training datasets, generated run outputs, or all trained model checkpoints.

## Running the current pipeline

The entrypoint currently uses paths such as input_videos/home3.mp4 and model files under models/. Add a video you have permission to analyze, obtain the required trained checkpoints, and update the paths in main.py before running it. The pipeline writes an annotated video to output_videos/.

Required files referenced by main.py include:

- models/player-detection-last.pt
- models/tracknet_model_best.pt
- models/ctb_regr_bounce.cbm
- models/keypoints_model.pth

Only the player detector and bounce-regression checkpoint are included. The TrackNet and court-keypoint checkpoints are larger model files and are not hosted in this repository. The exact dependency set for the complete pipeline also needs to be consolidated; TrackNet/requirements.txt documents the older TrackNet environment.

## Data and model files

Keep private match footage, datasets, cached detections, and generated videos out of source control. The .gitignore excludes common local data and model artifacts. Do not publish footage or model weights unless you have permission and have checked their licenses and size limits.

## Acknowledgments and references

- **Tennis Match Analysis** by [pratyusharoraa](https://github.com/pratyusharoraa/Tennis-Match-Analysis) — used as a reference for this project and its tennis court keypoint training resources. The court-keypoint training notebook uses the dataset link documented by that project.
- **TrackNet** by [yastrebksv](https://github.com/yastrebksv/TrackNet) — the TrackNet implementation in the TrackNet/ folder was adapted from this repository. TrackNet is described in [“A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications”](https://arxiv.org/abs/1907.03698).
  
- **Entrepreneurship Studio Program** summer research opportunity provided by Harvey Mudd College. I would like to especially think my teammates Bell Zhou and Emily Ing for helping me develop this program and the user interface. I would also like to thank Professor Zachary Dodds for advising us and guiding us through this project. Lastly, I would like to thank all the guest speakers and our peers from the lab for the insightful feedback and suggestions.  

## Current limitations

- The pipeline is a research prototype, not a packaged app or web service.
- Checkpoint and video paths are currently configured in source code.
- Court keypoints and ball/player detection can fail on footage that differs from the data used to train the models.
- The repository does not yet provide a unified install file or automated evaluation results.
