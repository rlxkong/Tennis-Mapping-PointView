import catboost as ctb
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import distance
import math

class BounceDetector:
    def __init__(self, model_path=None):
        self.model = ctb.CatBoostRegressor()
        self.threshold = 0.45
        if model_path:
            self.load_model(model_path)
        
    def load_model(self, path_model):
        self.model.load_model(path_model)
    
    def prepare_features(self, x_ball, y_ball):
        labels = pd.DataFrame({'frame': range(len(x_ball)), 'x-coordinate': x_ball, 'y-coordinate': y_ball})
        
        num = 3
        eps = 1e-15
        for i in range(1, num):
            labels['x_lag_{}'.format(i)] = labels['x-coordinate'].shift(i)
            labels['x_lag_inv_{}'.format(i)] = labels['x-coordinate'].shift(-i)
            labels['y_lag_{}'.format(i)] = labels['y-coordinate'].shift(i)
            labels['y_lag_inv_{}'.format(i)] = labels['y-coordinate'].shift(-i) 
            labels['x_diff_{}'.format(i)] = abs(labels['x_lag_{}'.format(i)] - labels['x-coordinate'])
            labels['y_diff_{}'.format(i)] = labels['y_lag_{}'.format(i)] - labels['y-coordinate']
            labels['x_diff_inv_{}'.format(i)] = abs(labels['x_lag_inv_{}'.format(i)] - labels['x-coordinate'])
            labels['y_diff_inv_{}'.format(i)] = labels['y_lag_inv_{}'.format(i)] - labels['y-coordinate']
            labels['x_div_{}'.format(i)] = abs(labels['x_diff_{}'.format(i)]/(labels['x_diff_inv_{}'.format(i)] + eps))
            labels['y_div_{}'.format(i)] = labels['y_diff_{}'.format(i)]/(labels['y_diff_inv_{}'.format(i)] + eps)

        for i in range(1, num):
            labels = labels[labels['x_lag_{}'.format(i)].notna()]
            labels = labels[labels['x_lag_inv_{}'.format(i)].notna()]
        labels = labels[labels['x-coordinate'].notna()] 
        
        colnames_x = ['x_diff_{}'.format(i) for i in range(1, num)] + \
                     ['x_diff_inv_{}'.format(i) for i in range(1, num)] + \
                     ['x_div_{}'.format(i) for i in range(1, num)]
        colnames_y = ['y_diff_{}'.format(i) for i in range(1, num)] + \
                     ['y_diff_inv_{}'.format(i) for i in range(1, num)] + \
                     ['y_div_{}'.format(i) for i in range(1, num)]
        colnames = colnames_x + colnames_y

        features = labels[colnames]
        return features, list(labels['frame'])
    
    def predict(self, ball_positions, smooth=True):
        x_ball = [pos[0] if pos else None for pos in ball_positions]
        y_ball = [pos[1] if pos else None for pos in ball_positions]

        if smooth:
            x_ball, y_ball = self.smooth_predictions(x_ball, y_ball)
        features, num_frames = self.prepare_features(x_ball, y_ball)
        preds = self.model.predict(features)
        ind_bounce = np.where(preds > self.threshold)[0]
        if len(ind_bounce) > 0:
            ind_bounce = self.postprocess(ind_bounce, preds)
        frames_bounce = [num_frames[x] for x in ind_bounce]
        return set(frames_bounce)
    
    def smooth_predictions(self, x_ball, y_ball):
        is_none = [int(x is None) for x in x_ball]
        interp = 5
        counter = 0
        for num in range(interp, len(x_ball)-1):
            if not x_ball[num] and sum(is_none[num-interp:num]) == 0 and counter < 3:
                x_ext, y_ext = self.extrapolate(x_ball[num-interp:num], y_ball[num-interp:num])
                x_ball[num] = x_ext
                y_ball[num] = y_ext
                is_none[num] = 0
                if x_ball[num+1]:
                    dist = distance.euclidean((x_ext, y_ext), (x_ball[num+1], y_ball[num+1]))
                    if dist > 80:
                        x_ball[num+1], y_ball[num+1], is_none[num+1] = None, None, 1
                counter += 1
            else:
                counter = 0  
        return x_ball, y_ball

    def extrapolate(self, x_coords, y_coords):
        xs = list(range(len(x_coords)))
        func_x = CubicSpline(xs, x_coords, bc_type='natural')
        x_ext = func_x(len(x_coords))
        func_y = CubicSpline(xs, y_coords, bc_type='natural')
        y_ext = func_y(len(x_coords))
        return float(x_ext), float(y_ext)    

    def postprocess(self, ind_bounce, preds):
        ind_bounce_filtered = [ind_bounce[0]]
        for i in range(1, len(ind_bounce)):
            if (ind_bounce[i] - ind_bounce[i-1]) != 1:
                cur_ind = ind_bounce[i]
                ind_bounce_filtered.append(cur_ind)
            elif preds[ind_bounce[i]] > preds[ind_bounce[i-1]]:
                ind_bounce_filtered[-1] = ind_bounce[i]
        return ind_bounce_filtered
    
    def get_hit_frames(self, bounce_frames, ball_positions):
        # Convert bounce_frames to list if it's a set
        bounce_list = list(bounce_frames) if isinstance(bounce_frames, set) else bounce_frames
        bounce_list = sorted([b for b in bounce_frames if isinstance(b, int)], key=int)

        hit_frames = []
        prev_valid_pos = None
        
        # First collect all valid positions with their frame indices
        valid_entries = []
        for frame_idx, pos in enumerate(ball_positions):
            if pos is not None and isinstance(pos, (tuple, list)) and len(pos) == 2:
                try:
                    x, y = float(pos[0]), float(pos[1])
                    valid_entries.append((frame_idx, x, y))
                except (TypeError, ValueError):
                    continue  # Skip invalid numeric values
        
        # Simplified hit detection (no player tracking needed)
        for i in range(1, len(valid_entries)):
            prev_frame, prev_x, prev_y = valid_entries[i-1]
            curr_frame, curr_x, curr_y = valid_entries[i]
            
            # Calculate motion characteristics
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            speed = (dx**2 + dy**2)**0.5
            
            # Hit detection conditions (simplified)
            if i < len(valid_entries) - 1:
                next_frame, next_x, next_y = valid_entries[i+1]
                next_dx = next_x - curr_x
                next_dy = next_y - curr_y
                next_speed = (next_dx**2 + next_dy**2)**0.5
                
                # Condition: Sudden speed/direction change
                speed_change = abs(next_speed - speed)
                angle_change = abs(math.atan2(dy, dx) - math.atan2(next_dy, next_dx))
                
                is_near_bounce = any(abs(curr_frame - bounce) <= 10 for bounce in bounce_list)  # +/-2 frame tolerance
                if speed_change > 30 and abs(angle_change) > 0.5 and not is_near_bounce:  # Adjust thresholds
                    hit_frames.append(curr_frame-3)

            filtered_hit_frames = self.filter_close_hits(hit_frames, min_gap=10)
        
        # # Combine with bounce frames
        # combined_frames = list(set(bounce_list + filtered_hit_frames))
        # combined_frames.sort()
        
        print(f"Detected hits: {filtered_hit_frames}")
        return filtered_hit_frames
    
    def filter_close_hits(self, hit_frames, min_gap=10):
        if not hit_frames:
            return []
        
        filtered = [hit_frames[0]]  # Keep the first hit
        
        for current_hit in hit_frames[1:]:
            if current_hit - filtered[-1] > min_gap:
                filtered.append(current_hit)
        
        return filtered