from .model import BallTrackerNet
import torch
import cv2
from .general import postprocess
from tqdm import tqdm
import numpy as np
from itertools import groupby
from scipy.spatial import distance
import sys
import pickle
sys.path.append('../')

class TracksNet:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = BallTrackerNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def read_video(self, path_video):
        """ Read video file    
        :params
            path_video: path to video file
        :return
            frames: list of video frames
            fps: frames per second
        """
        cap = cv2.VideoCapture(path_video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (1280, 720))
                frames.append(frame)
            else:
                break
        cap.release()
        return frames, fps

    def infer_model(self, frames, read_from_stub=False, stub_path=None):
        """ Run pretrained model on a consecutive list of frames    
        :params
            frames: list of consecutive video frames
            model: pretrained model
        :return    
            ball_track: list of detected ball points
            dists: list of euclidean distances between two neighbouring ball points
        """

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        height = 360
        width = 640

        # Get original frame dimensions for scaling
        orig_height, orig_width = frames[0].shape[:2]
        scale_x = orig_width / width
        scale_y = orig_height / height

        dists = [-1]*2
        ball_track = [(None,None)]*2
        for num in tqdm(range(2, len(frames))):
            img = cv2.resize(frames[num], (640, 360))
            img_prev = cv2.resize(frames[num - 1], (640, 360))
            img_preprev = cv2.resize(frames[num - 2], (640, 360))
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32)/255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            out = self.model(torch.from_numpy(inp).float().to(self.device))
            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = postprocess(output)
            ball_track.append((x_pred, y_pred))

            # Scale coordinates back to original frame size
            if x_pred is not None and y_pred is not None:
                x_pred = int(x_pred * scale_x)
                y_pred = int(y_pred * scale_y)

            if ball_track[-1][0] and ball_track[-2][0]:
                dist = distance.euclidean(ball_track[-1], ball_track[-2])
            else:
                dist = -1
            dists.append(dist)  

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump((ball_track, dists), f)

        return ball_track, dists 

    def remove_outliers(self, ball_track, dists, max_dist = 100):
        """ Remove outliers from model prediction    
        :params
            ball_track: list of detected ball points
            dists: list of euclidean distances between two neighbouring ball points
            max_dist: maximum distance between two neighbouring ball points
        :return
            ball_track: list of ball points
        """
        outliers = list(np.where(np.array(dists) > max_dist)[0])
        for i in outliers:
            if (dists[i+1] > max_dist) | (dists[i+1] == -1):       
                ball_track[i] = (None, None)
                outliers.remove(i)
            elif dists[i-1] == -1:
                ball_track[i-1] = (None, None)
        return ball_track  

    def split_track(self, ball_track, max_gap=4, max_dist_gap=80, min_track=5):
        """ Split ball track into several subtracks in each of which we will perform
        ball interpolation.    
        :params
            ball_track: list of detected ball points
            max_gap: maximun number of coherent None values for interpolation  
            max_dist_gap: maximum distance at which neighboring points remain in one subtrack
            min_track: minimum number of frames in each subtrack    
        :return
            result: list of subtrack indexes    
        """
        list_det = [0 if x[0] else 1 for x in ball_track]
        groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

        cursor = 0
        min_value = 0
        result = []
        for i, (k, l) in enumerate(groups):
            if (k == 1) & (i > 0) & (i < len(groups) - 1):
                dist = distance.euclidean(ball_track[cursor-1], ball_track[cursor+l])
                if (l >=max_gap) | (dist/l > max_dist_gap):
                    if cursor - min_value > min_track:
                        result.append([min_value, cursor])
                        min_value = cursor + l - 1        
            cursor += l
        if len(list_det) - min_value > min_track: 
            result.append([min_value, len(list_det)]) 
        return result    

    def interpolation(self, coords):
        """ Run ball interpolation in one subtrack    
        :params
            coords: list of ball coordinates of one subtrack    
        :return
            track: list of interpolated ball coordinates of one subtrack
        """
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
        y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

        nons, yy = nan_helper(x)
        x[nons]= np.interp(yy(nons), yy(~nons), x[~nons])
        nans, xx = nan_helper(y)
        y[nans]= np.interp(xx(nans), xx(~nans), y[~nans])

        track = [*zip(x,y)]
        return track

    def draw_bbox(self, frames, ball_track, trace=7):
        """ Draw ball track circles on frames
        :params
            frames: list of video frames (modified in-place)
            ball_track: list of (x, y) positions of ball
            trace: number of previous frames to show ball trace
        :return
            frames: list of frames with ball trace drawn
        """
        output_frames = []
        frame_height, frame_width = frames[0].shape[:2]
        scale_x = frame_width/1280
        scale_y = frame_height/720

        for num in range(len(frames)):
            frame = frames[num].copy()
            for i in range(trace):
                if num - i > 0 and ball_track[num - i][0] is not None:
                    x = int(ball_track[num - i][0] * scale_x)
                    y = int(ball_track[num - i][1] * scale_y)
                    frame = cv2.circle(frame, (x, y), radius=0, color=(0, 255, 255), thickness=10 - i)
            output_frames.append(frame)
        return output_frames
    
    def track(self, video_path, read_from_stub=False, stub_path=None):
        frames, fps = self.read_video(video_path)
        ball_track, dists = self.infer_model(frames,
                                             read_from_stub=read_from_stub,
                                            stub_path=stub_path)
        ball_track = self.remove_outliers(ball_track, dists)

        subtracks = self.split_track(ball_track)
        for r in subtracks:
            ball_subtrack = ball_track[r[0]:r[1]]
            ball_subtrack = self.interpolation(ball_subtrack)
            ball_track[r[0]:r[1]] = ball_subtrack
            
        return ball_track    
        
        
        
        
        
