import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import PchipInterpolator
from tabulate import tabulate

# mostly working, sometimes detects false bounces or see hit as a bounce 
def remove_coordinate_outliers(data, z_thresh=2.5):
    coords = np.array([[x, y] for _, x, y in data if not (x == 0 and y == 0)])
    if len(coords) < 5:
        return data

    mean = np.mean(coords, axis=0)
    std = np.std(coords, axis=0)
    z_scores = np.abs((coords - mean) / std)
    keep_indices = np.all(z_scores < z_thresh, axis=1)

    filtered_data = []
    j = 0
    for frame, x, y in data:
        if not (x == 0 and y == 0):
            if keep_indices[j]:
                filtered_data.append([frame, x, y])
            j += 1
        else:
            filtered_data.append([frame, x, y])

    return filtered_data

def interpolate_missing_points(data, method='linear'):
    valid_data = [(frame, x, y) for frame, x, y in data if not (x == 0 and y == 0)]
    if len(valid_data) < 3:
        return data

    frames = np.array([frame for frame, _, _ in valid_data])
    x_coords = np.array([x for _, x, _ in valid_data])
    y_coords = np.array([y for _, _, y in valid_data])

    f_x = PchipInterpolator(frames, x_coords)
    f_y = PchipInterpolator(frames, y_coords)

    all_frames = list(range(int(frames.min()), int(frames.max()) + 1))
    interpolated_data = []

    for frame in all_frames:
        if frame in frames:
            idx = np.where(frames == frame)[0][0]
            interpolated_data.append([frame, x_coords[idx], y_coords[idx]])
        else:
            interpolated_data.append([frame, float(f_x(frame)), float(f_y(frame))])

    return interpolated_data

# def remove_missing_points(data):
#     """
#     Removes data points with missing (0, 0) coordinates.
#     """
#     return [(frame, x, y) for frame, x, y in data if not (x == 0 and y == 0)]


def detect_local_maxima_bounces(data, distance=20, prominence=15, max_x_velocity=5.0):
    valid_data = [(f, x, y) for f, x, y in data if not (x == 0 and y == 0)]
    if len(valid_data) < 5:
        return []

    frames = np.array([f for f, _, _ in valid_data])
    x_vals = np.array([x for _, x, _ in valid_data])
    y_vals = np.array([y for _, _, y in valid_data])

    window = min(7, len(y_vals) if len(y_vals) % 2 == 1 else len(y_vals)-1)
    x_smooth = savgol_filter(x_vals, window, polyorder=2)
    y_smooth = savgol_filter(y_vals, window, polyorder=2)

    peaks, _ = find_peaks(y_smooth, distance=distance, prominence=prominence)

    x_velocity = np.gradient(x_smooth)

    bounces = [(frames[i], valid_data[i][1], valid_data[i][2]) for i in peaks]
    return bounces

def detect_bounces_by_slope_sign_change(data, distance=20):
    """
    Detect bounces by looking for ANY velocity sign change.
    This finds both peaks and valleys in the trajectory.
    """
    valid_data = [(f, x, y) for f, x, y in data if not (x == 0 and y == 0)]
    if len(valid_data) < 5:
        return []

    frames = np.array([f for f, _, _ in valid_data])
    y_vals = np.array([y for _, _, y in valid_data])

    # Smooth y to reduce noise
    window = min(7, len(y_vals) if len(y_vals) % 2 == 1 else len(y_vals) - 1)
    y_smooth = savgol_filter(y_vals, window, polyorder=2)

    # Compute velocity (first derivative)
    velocity = np.gradient(y_smooth)

    bounces = []
    for i in range(1, len(velocity)):
        # Check for ANY velocity sign change
        sign_change = ((velocity[i-1] > 0 and velocity[i] <= 0) or  # positive to negative
                      (velocity[i-1] < 0 and velocity[i] >= 0))     # negative to positive
        
        if sign_change:
            # Enforce minimum distance between bounces
            if len(bounces) == 0 or (frames[i] - bounces[-1][0]) >= distance:
                bounces.append((frames[i], valid_data[i][1], valid_data[i][2]))

    return bounces


def detect_bounces_velocity_threshold(data, min_frames_between_bounces=50, velocity_threshold=3.0):
    bounces = []
    valid_data = [(frame, x, y) for frame, x, y in data if not (x == 0 and y == 0)]
    if len(valid_data) < 3:
        return bounces

    # Calculate vertical velocities (y differences)
    velocities = [valid_data[i][2] - valid_data[i - 1][2] for i in range(1, len(valid_data))]

    if len(velocities) > 5:
        smoothed_velocities = savgol_filter(velocities, 5, 2)
    else:
        smoothed_velocities = velocities

    last_bounce_frame = -min_frames_between_bounces

    for i in range(1, len(smoothed_velocities)):
        current_frame = valid_data[i + 1][0]
        prev_vel = smoothed_velocities[i - 1]
        curr_vel = smoothed_velocities[i]

        if (prev_vel > velocity_threshold and curr_vel < -velocity_threshold):
            if (current_frame - last_bounce_frame) >= min_frames_between_bounces:
                if abs(curr_vel - prev_vel) > velocity_threshold * 2.0:
                    bounces.append((current_frame, valid_data[i + 1][1], valid_data[i + 1][2]))
                    last_bounce_frame = current_frame

    return bounces


def combine_bounce_detections(bounces1, bounces2, frame_tolerance=20):
    combined = []
    used_indices = set()

    for i1, (f1, x1, y1) in enumerate(bounces1):
        for i2, (f2, x2, y2) in enumerate(bounces2):
            if i2 in used_indices:
                continue
            if abs(f1 - f2) <= frame_tolerance:
                avg_frame = int(round((f1 + f2) / 2))
                avg_x = (x1 + x2) / 2
                avg_y = (y1 + y2) / 2
                combined.append((avg_frame, avg_x, avg_y))
                used_indices.add(i2)
                break  # only take first match from bounces2

    combined.sort(key=lambda x: x[0])
    return combined

def visualize_analysis(data, bounce_groups, labels):
    import matplotlib.pyplot as plt
    import numpy as np

    frames = np.array([f for f, x, y in data])
    x_vals = np.array([x for f, x, y in data])
    y_vals = np.array([y for f, x, y in data])

    # Compute simple velocity: delta x / delta frame
    frame_diffs = np.diff(frames)
    x_diffs = np.diff(x_vals)
    y_diffs = np.diff(y_vals)

    x_vel = x_diffs / frame_diffs
    y_vel = y_diffs / frame_diffs
    mid_frames = frames[:-1]  # Align diff output with frame indices

    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # --- Position plot ---
    axs[0].plot(frames, y_vals, label="Ball Y Position", color='blue', linewidth=2)
    axs[0].plot(frames, x_vals, label="Ball X Position", color='purple', linewidth=2)

    colors = ['orange', 'green', 'red']
    for bounces, label, color in zip(bounce_groups, labels, colors):
        if bounces:
            bounce_frames = [f for f, x, y in bounces]
            bounce_y_vals = [y for f, x, y in bounces]
            axs[0].scatter(bounce_frames, bounce_y_vals, label=label, color=color, s=60, marker='x')

    axs[0].set_title("Ball Position vs Frame")
    axs[0].set_ylabel("Position (X and Y)")
    axs[0].legend()
    axs[0].grid(True)

    # --- Velocity plot ---
    axs[1].plot(mid_frames, y_vel, label="Y Velocity (dy/dF)", color='blue', linestyle='--')
    axs[1].plot(mid_frames, x_vel, label="X Velocity (dx/dF)", color='purple', linestyle='--')
    axs[1].set_title("Ball Velocity (First Difference of Position)")
    axs[1].set_xlabel("Frame")
    axs[1].set_ylabel("Velocity")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def print_velocity_table(data):
    

    frames = np.array([f for f, x, y in data])
    x_vals = np.array([x for f, x, y in data])
    y_vals = np.array([y for f, x, y in data])

    # Compute velocity as first difference
    frame_diffs = np.diff(frames)
    x_diffs = np.diff(x_vals)
    y_diffs = np.diff(y_vals)

    x_vel = x_diffs / frame_diffs
    y_vel = y_diffs / frame_diffs
    mid_frames = frames[:-1]  # Corresponding frame numbers for velocity rows

    # Build table
    table = []
    for f, xv, yv in zip(mid_frames, x_vel, y_vel):
        table.append([f, round(xv, 2), round(yv, 2)])

    headers = ["Frame", "X Velocity", "Y Velocity"]
    print(tabulate(table, headers=headers, tablefmt="grid"))




def main():
    try:
        with open("output/video2_coordinates.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: Could not find output/match11_coordinates.json")
        return

    print(f"Loaded {len(data)} data points")

    filtered_data = remove_coordinate_outliers(data, z_thresh=2.5)
    print(f"After outlier removal: {len(filtered_data)} data points")

    interpolated_data = interpolate_missing_points(filtered_data)
    print(f"After removing missing points: {len(interpolated_data)} data points")

    # Step 1: Detect toss (first bounce)
    all_bounces = detect_bounces_velocity_threshold(
        interpolated_data,
        min_frames_between_bounces=50,
        velocity_threshold=2.5
    )

    if not all_bounces:
        print("No toss (first bounce) detected.")
        return

    first_bounce_frame = all_bounces[0][0]
    print(f"Detected toss at frame {first_bounce_frame}. Keeping toss and detecting all bounces...")

    # Step 2: Keep all data including toss
    trimmed_data = interpolated_data  # no removal of frames before toss

    # Step 3: Run both detection methods on full data
    bounces_maxima = detect_local_maxima_bounces(trimmed_data, distance=10, prominence=5)
    bounces_direction = detect_bounces_by_slope_sign_change(trimmed_data, distance=10)

    # Step 4: Combine bounces
    combined_bounces = combine_bounce_detections(bounces_maxima, bounces_direction, frame_tolerance=5)

    # Step 5: Display results
    print("\n=== BOUNCE DETECTION RESULTS ===")
    print(f"Local Maxima (find_peaks): {len(bounces_maxima)} bounces")
    for f, x, y in bounces_maxima:
        print(f"  Frame {f}: ({x:.1f}, {y:.1f})")

    print(f"Direction Change Maxima: {len(bounces_direction)} bounces")
    for f, x, y in bounces_direction:
        print(f"  Frame {f}: ({x:.1f}, {y:.1f})")

    print(f"Combined Bounces: {len(combined_bounces)} bounces")
    for f, x, y in combined_bounces:
        print(f"  Frame {f}: ({x:.1f}, {y:.1f})")

    visualize_analysis(
        trimmed_data,
        [bounces_maxima, bounces_direction, combined_bounces],
        ['Local Maxima', 'Direction Change', 'Combined']
    )
    print_velocity_table(trimmed_data)

    # Save combined bounces to JSON
    output_path = "output/video2_bounces.json"
    with open(output_path, "w") as out_file:
        json.dump(combined_bounces, out_file, indent=2)

    print(f"\nSaved combined bounce detections to {output_path}")

if __name__ == "__main__":
    main()