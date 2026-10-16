# PointView ESP: Project Roadmap

This roadmap captures a possible path from the project concept to an evaluated product. It is a planning document, not a record of completed work.

## 1. Validate the problem

- Interview tennis players and coaches about how they review match footage today.
- Identify the most valuable first use case and what “affordable” means to the intended users.
- Test whether players want automated stats, searchable notes, faster video review, or a combination.

## 2. Define a narrow first version

- Select the first supported video source and upload flow.
- Decide whether inactive-play trimming is manual, automatic, or assisted.
- Define a small set of useful, explainable tennis stats.
- Sketch how a player finds a point, records a note, and returns to it later.

## 3. Evaluate the video pipeline

- Establish consent, privacy, storage, and deletion rules before collecting footage.
- Label a small evaluation set with clear definitions for players, ball, court, and events.
- Compare player tracking (YOLOv8 with BYTETrack), ball tracking (TrackNet), and court mapping (keypoints with homography) on representative footage.
- Record accuracy, failure cases, runtime, and hardware needs before choosing an approach.

## 4. Build and test with users

- Create a minimal review experience around real, consented footage.
- Let users inspect, correct, or dismiss uncertain analysis.
- Test the workflow with players and coaches and prioritize changes from observed use.

## 5. Prepare for a public release

- Document setup, supported formats, known limitations, and data handling.
- Check dependencies, model licenses, and rights to any sample footage.
- Add a demo and screenshots only after a working implementation is available.

## Open decisions

- Which player segment and platform should the first version serve?
- What is the smallest useful set of stats?
- Where will video processing happen, and how long will uploads be retained?
- What quality bar must tracking meet before results are shown?
- Which license and contribution policy fit the founders' plans?
