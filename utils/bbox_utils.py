def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2)/2)
    center_y = int((y1 + y2)/2)
    return (center_x, center_y)

def measure_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def get_foot_position_from_bbox(bbox):
    """
    Returns the foot position from a bounding box.
    The foot position is assumed to be at the bottom center of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    foot_x = int((x1 + x2) / 2)
    foot_y = y2  # Assuming the foot is at the bottom of the bounding box
    return (foot_x, foot_y)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float('inf')
    keypoint_ind = keypoint_indices[0]  # Default to the first keypoint index
    for keypoint_index in keypoint_indices:
        keypoint = keypoints[keypoint_index * 2], keypoints[keypoint_index * 2 + 1]
        distance = abs(point[1]-keypoint[1])  # Only consider the y-coordinate for distance
        # If the distance is less than the closest distance found so far, update
        if distance < closest_distance:
            closest_distance = distance
            keypoint_ind = keypoint_index
            
    return keypoint_ind