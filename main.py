from utils import (read_video, 
                   save_video)
from trackers import PlayerTracker#, BallTracker
#from court_line_detector import CourtLineDetector
from tennis_court_keypoint_detector import LabelKeypoints, CourtKeypointsDrawer
from TrackNet import TracksNet
#from Predict import GridTracksNet
from mini_court import MiniCourt
from bounce_detector import BounceDetector
import cv2

def main():
    #read video
    #retrieving the video [input_video is where you put your vid]
    input_video_path = "input_videos/home3.mp4"
    video_frames = read_video(input_video_path)

    #detect players and ball --> initialize then detect
    player_tracker = PlayerTracker(model_path='models/player-detection-last.pt')
    #ball_tracker = BallTracker(model_path='models/best.pt')
    ball_tracker = TracksNet(model_path='models/tracknet_model_best.pt', device = 'cpu')
    #ball_tracker = GridTracksNet(model_weights_path='models/predict_gridTrackNet_weights.h5')
    bounce_detector = BounceDetector(model_path='models/ctb_regr_bounce.cbm')
    #MiniCourt Initialize
    minicourt = MiniCourt(video_frames[0])

    #Initialize keypoints
    court_keypoints = LabelKeypoints(image=video_frames[0], output_path="keypoints_pkl/court_keypoints.pkl")
    #Get Court Keypoints
    court_keypoint_drawer = CourtKeypointsDrawer()

    #read_from_stubs if ran before and is true will return the result much faster
    #comes from the last video read
    player_detections = player_tracker.detect_frames(video_frames, 
                                                     read_from_stub=False,
                                                     stub_path="tracker_stubs/player_detections.pkl")
    # ball_detections = ball_tracker.detect_frames(video_frames,
    #                                             read_from_stub=True,
    #                                             stub_path = "tracker_stubs/ball_detections.pkl")
    # ball_detections = ball_tracker.interpolate_ball_position(ball_detections)
    ball_track = ball_tracker.track(input_video_path,
                                    read_from_stub=False,
                                    stub_path="tracker_stubs/ball_detections.pkl")


    #Court line detector model
    court_model_path = "models/keypoints_model.pth"
    #court_line_detector = CourtLineDetector(court_model_path)
    #court_keypoints = court_line_detector.predict(video_frames[0])
    get_court_keypoints = court_keypoints.run(save_to_files=False)

    get_bounce = bounce_detector.predict(ball_track, smooth=True)
    get_hit = bounce_detector.get_hit_frames(get_bounce, ball_track)
    print(f"List of frames of bounces: {get_bounce}")

    bounce_ball_positions = minicourt.get_frames_with_hits(ball_track, get_bounce, get_hit)
    #print(f"List of frames of bounces: {bounce_ball_positions}")

    player_positions, ball_positions = minicourt.map_detections_to_mini_court(player_detections, bounce_ball_positions, get_court_keypoints)
    keypoints_per_frame = [get_court_keypoints] * len(video_frames)
    

    #Choose players
    player_detections = player_tracker.choose_and_filter_players(get_court_keypoints, player_detections)

    output_video_frames = minicourt.draw_mini_court_with_detections(video_frames, player_positions, ball_positions)

    ##Draw Player and Ball Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(output_video_frames, player_detections)
    #output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = ball_tracker.draw_bbox(output_video_frames, ball_track, trace=7)

    ##Draw court Keypoints
    # output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    output_video_frames = court_keypoint_drawer.draw(output_video_frames, keypoints_per_frame)

    ##Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    #saves videos into output
    save_video(input_video_path, output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()