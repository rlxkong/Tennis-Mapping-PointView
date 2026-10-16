import cv2

def read_video(video_path):
    """ Takes a video path and reutrn"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        # when ret is false (no more frames) loop ends
        ret, frame = cap.read()
        if not ret:
            break 
        frames.append(frame)
    cap.release()
    return frames

def save_video(video_path, output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()