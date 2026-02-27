#!/usr/bin/env python3
"""
Badminton Pose Extraction Pipeline
Extracts pose keypoints and player positions from badminton match videos.

This script:
1. Downloads the match video
2. Extracts frames for each stroke
3. Runs YOLOv8 pose estimation
4. Merges detections to exactly 2 players
5. Saves poses, positions, and labels as .pkl files

Usage:
    python badminton_pose_pipeline.py --match_index 21
    python badminton_pose_pipeline.py -m 21 --visualize
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import ast
import pickle

import pandas as pd
import numpy as np
import cv2
import torch
from ultralytics import YOLO


# def install_dependencies():
#     """Install required packages if not already installed."""
#     print("=" * 60)
#     print("INSTALLING DEPENDENCIES")
#     print("=" * 60)
#
#     packages = [
#         "ultralytics",
#         "yt-dlp",
#         "pandas",
#         "numpy",
#         "opencv-python",
#         "matplotlib",
#         "torch",
#         "torchvision"
#     ]
#
#     for package in packages:
#         try:
#             # Special handling for package names vs import names
#             import_name = package.replace("-", "_")
#             if package == "opencv-python":
#                 import_name = "cv2"
#             __import__(import_name)
#             print(f"✓ {package} already installed")
#         except ImportError:
#             print(f"Installing {package}...")
#             subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
#             print(f"✓ {package} installed")
#
#     # Clone the repository if it doesn't exist
#     repo_path = Path("CoachAI-Projects")
#     if not repo_path.exists():
#         print("\nCloning CoachAI-Projects repository...")
#         subprocess.check_call(["git", "clone", "https://github.com/wywyWang/CoachAI-Projects/"])
#         print("✓ Repository cloned")
#     else:
#         print("✓ CoachAI-Projects repository already exists")
#
#     print("✓ All dependencies ready\n")


def print_cwd_size():
    total_size = 0
    cwd = os.getcwd()

    for dirpath, dirnames, filenames in os.walk(cwd):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if broken symlink
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    print(f"Current working directory: {cwd}")
    print(f"Total size: {total_size / (1024 * 1024):.2f} MB")
def download_video(url, filename):
    """Download video using yt-dlp."""
    if Path(filename).exists():
        print(f"✓ Video already exists: {filename}")
        return

    print(f"Downloading video: {filename}")

    cmd = [
        "yt-dlp",
        "-f", "136",
        "--remote-components", "ejs:github",
        "-o", filename,
        "--cookies", "cookies.txt",
        url
    ]
    try:
        subprocess.check_call(cmd)
        print(f"✓ Video downloaded: {filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video: {e}")
        sys.exit(1)

def pixel_to_real(H, main_pixel):
    """Convert pixel coordinates to real court coordinates using homography matrix."""
    x, y = main_pixel
    pixel_h = np.array([x, y, 1.0])

    # Multiply by inverse of H
    real_h = np.linalg.inv(H) @ pixel_h

    # Normalize
    real_h /= real_h[2]

    return [real_h[0], real_h[1]]


def extract_stroke_frames(video_path, df, index, match_name, set_name, output_base_dir,
                          window_before=15, window_after=14):
    """
    Extract 30 frames for one stroke and save them to disk.

    Returns:
        tuple: (output_dir, main_label) or (None, None) if failed
    """
    output_dir = output_base_dir / match_name / set_name / str(index)
    os.makedirs(output_dir, exist_ok=True)

    stroke_row = df.iloc[index]

    # Extract stroke metadata from CSV
    rally_num = int(stroke_row['rally'])
    ball_round = int(stroke_row['ball_round'])
    stroke_type = stroke_row['type']
    player = stroke_row['player']
    hit_frame = int(stroke_row['frame_num'])

    # Calculate frame range
    start_frame = hit_frame - window_before
    end_frame = hit_frame + window_after

    # Create main_label array
    main_label = [
        stroke_type,  # 0: shot/stroke type (string)
        rally_num,  # 1: rally number in the set (int)
        ball_round,  # 2: ball number i.e. stroke order in rally (int)
        match_name,  # 3: match name (string)
        set_name,  # 4: set name (string) e.g. 'set1'
        player,  # 5: player hitting ID (string) 'A' or 'B'
        start_frame,  # 6: starting_frame stroke (int/frame)
        end_frame,  # 7: end_frame stroke (int/frame)
        hit_frame  # 8: hit_frame (int/frame)
    ]

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Adjust frame range if it goes out of bounds
    actual_start = max(0, start_frame)
    actual_end = min(total_frames - 1, end_frame)

    # Extract frames
    frames_extracted = 0

    for frame_idx in range(actual_start, actual_end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            frame_filename = output_dir / f"frame_{frame_idx:03d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            frames_extracted += 1

    cap.release()

    if frames_extracted == 0:
        return None, None

    return output_dir, main_label


def merge_keypoints(kp_list):
    """Merge multiple skeletons: keep highest-confidence keypoint for every joint."""
    merged = np.zeros((17, 3))
    for kp in kp_list:
        for i in range(17):
            if kp[i, 2] > merged[i, 2]:
                merged[i] = kp[i]
    return merged


def merge_to_two_players(kps):
    """Cluster ≥3 detections into exactly TWO players using horizontal centroid."""
    centers = []
    for i in range(len(kps)):
        valid = kps[i, :, 2] > 0
        if np.sum(valid) == 0:
            continue
        mean_x = np.mean(kps[i, valid][:, 0])
        centers.append((i, mean_x))

    if not centers:
        return None

    centers = sorted(centers, key=lambda x: x[1])

    cutoff = (centers[0][1] + centers[-1][1]) / 2
    left_group = [i for (i, mx) in centers if mx <= cutoff]
    right_group = [i for (i, mx) in centers if mx > cutoff]

    if len(right_group) == 0:
        right_group = [left_group.pop()]

    left_kp = merge_keypoints([kps[i] for i in left_group])
    right_kp = merge_keypoints([kps[i] for i in right_group])

    return np.stack([left_kp, right_kp])  # (2,17,3)


def extract_pixel_poses_with_ankle_mean(frame_path, pose_model, H, roi=None,
                                        conf_thresh=0.3, device='cpu', visualize=False):
    """
    Extract pose keypoints and ankle positions from a frame.

    Returns:
        tuple: (clean_kps, mean_ankles_real) or (None, None) if failed
            - clean_kps: (2, 17, 3) array of keypoints for 2 players
            - mean_ankles_real: (2, 2) array of ankle positions in real-world coords
    """
    img = cv2.imread(frame_path)
    if img is None:
        raise ValueError(f"Cannot load {frame_path}")

    # Optional ROI crop
    if roi is not None:
        xs = [int(p[0]) for p in roi]
        ys = [int(p[1]) for p in roi]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        img = img[y1 - 100:y2, x1:x2]

    # YOLO inference
    results = pose_model(img, device=device, verbose=False)
    kps = results[0].keypoints.data.cpu().numpy()

    # Remove low-confidence keypoints
    kps[kps[:, :, 2] < conf_thresh] = 0
    n = len(kps)

    # CASE 1: < 2 players → reject
    if n < 2:
        return None, None

    # CASE 2: exactly 2 players → clean
    if n == 2:
        clean_kps = kps
    else:
        # CASE 3: ≥3 players → merge into TWO
        clean_kps = merge_to_two_players(kps)
        if clean_kps is None:
            return None, None

    # Compute mean ankle (pixel space)
    mean_ankles_px = np.zeros((2, 2))
    for pid in range(2):
        L = clean_kps[pid, 15]
        R = clean_kps[pid, 16]
        pts = [p[:2] for p in [L, R] if p[2] > 0]
        mean_ankles_px[pid] = np.mean(pts, axis=0) if pts else [0, 0]

    # Convert to real-world coords
    mean_ankles_real = np.array([pixel_to_real(H, p) for p in mean_ankles_px],
                                dtype=np.float64)

    # Optional visualization
    if visualize:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(16, 12))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            colors = ["red", "blue"]

            for pid in range(2):
                for k in range(17):
                    x, y, c = clean_kps[pid, k]
                    if c > 0:
                        plt.scatter(x, y, color=colors[pid], s=40)
                        plt.text(x + 3, y - 3, str(k), color="white", fontsize=8)

                ax, ay = mean_ankles_px[pid]
                plt.scatter(ax, ay, color=colors[pid], s=120, marker="X")

            plt.title("Merged pose detections → always 2 players")
            plt.axis("off")
            plt.show()
        except Exception as e:
            print(f"Visualization failed: {e}")

    return clean_kps, mean_ankles_real


def process_strokes_to_pkl(match_name, set_dirs, dfs, pose_model, H, device,
                           output_base_dir, visualize=False):
    """
    Process all strokes and generate pkl files.

    Returns:
        tuple: (all_poses, all_positions, all_labels, skip_count)
    """
    all_poses = []
    all_positions = []
    all_labels = []
    skip_count = 0
    total_strokes = sum(len(df) for df in dfs)
    processed = 0

    print("\n" + "=" * 60)
    print("PROCESSING STROKES WITH POSE ESTIMATION")
    print("=" * 60)

    for set_name, df in zip(set_dirs, dfs):
        set_dir = output_base_dir / match_name / set_name

        if not set_dir.exists():
            print(f"Warning: {set_dir} does not exist, skipping")
            continue

        print(f"\nProcessing {set_name} ({len(df)} strokes)...")

        for stroke_no in range(len(df)):
            stroke_dir = set_dir / str(stroke_no)



            frames = sorted([f for f in os.listdir(stroke_dir) if f.endswith('.jpg')])



            pose_per_stroke = []
            positions_per_stroke = []
            label_per_stroke = df.iloc[stroke_no]['main_label']
            skip_stroke = False

            for frame in frames:
                frame_path = str(stroke_dir / frame)
                kp, ankles = extract_pixel_poses_with_ankle_mean(
                    frame_path, pose_model, H, roi=None,
                    device=device, visualize=visualize
                )

                if kp is None or ankles is None:
                    print("_________________did not detect 2 people")
                    skip_stroke = True
                    break

                pose_per_stroke.append(kp)
                positions_per_stroke.append(ankles)

            if skip_stroke:
                skip_count += 1
                processed += 1
                print(f"_______________________________________skipping the stroke {stroke_dir}=====> current skip count{skip_count}")
                continue

            # Stack arrays
            poses_np = np.stack(pose_per_stroke)  # (30, 2, 17, 3)
            positions_np = np.stack(positions_per_stroke)  # (30, 2, 2)
            positions_np_rearranged = positions_np.transpose(1, 0, 2)  # (2, 30, 2)

            all_poses.append(poses_np)
            all_positions.append(positions_np_rearranged)
            all_labels.append(label_per_stroke)

            processed += 1
            print(f"  ✓ Processed {set_name}/{stroke_no} ({processed}/{total_strokes}) skipped {skip_count}")

    return all_poses, all_positions, all_labels, skip_count

def clone_repo_if_needed():
    """Clone CoachAI-Projects repository if it doesn't exist."""
    repo_path = Path("CoachAI-Projects")
    if not repo_path.exists():
        print("Cloning CoachAI-Projects repository...")
        subprocess.check_call(["git", "clone", "https://github.com/wywyWang/CoachAI-Projects/"])
        print("✓ Repository cloned")
    else:
        print("✓ CoachAI-Projects repository already exists")


def cleanup_generated_files(match_name, keep_pkl=True):
    """
    Delete all generated files and folders except PKL files.

    Args:
        match_name (str): Name of the match to clean up
        keep_pkl (bool): If True, keep .pkl files
    """
    print("\n" + "=" * 60)
    print("CLEANING UP GENERATED FILES")
    print("=" * 60)

    import shutil

    # Delete video file
    video_file = Path(f"{match_name}.mp4")
    if video_file.exists():
        video_file.unlink()
        print(f"✓ Deleted video: {video_file}")

    # Delete extracted frames directory
    frames_dir = Path(match_name)
    if frames_dir.exists() and frames_dir.is_dir():
        shutil.rmtree(frames_dir)
        print(f"✓ Deleted frames directory: {frames_dir}")

    # Optionally delete PKL files
    if not keep_pkl:
        for pkl_file in Path.cwd().glob("*.pkl"):
            pkl_file.unlink()
            print(f"✓ Deleted: {pkl_file}")

    print("✓ Cleanup complete")
def main(match_index, visualize=False):
    """
    Main pipeline function.

    Args:
        match_index (int): Index of the match in match.csv
        skip_install (bool): Skip dependency installation
        visualize (bool): Show pose visualization plots
    """
    # Install dependencies


    # Set up paths

    # clone_repo_if_needed()
    root = Path("CoachAI-Projects/ShuttleSet/set")
    if not root.exists():
        clone_repo_if_needed()
    output_base_dir = Path.cwd()

    # Load match data
    print("\n" + "=" * 60)
    print("LOADING MATCH DATA")
    print("=" * 60)

    match_df = pd.read_csv(root / 'match.csv')
    homography_df = pd.read_csv(root / 'homography.csv')

    # Get match information
    if match_index >= len(match_df):
        print(f"Error: Match index {match_index} is out of range. Total matches: {len(match_df)}")
        sys.exit(1)

    name = match_df.iloc[match_index]['video']
    url = match_df.iloc[match_index]['url']

    print(f"Match: {name}")
    print(f"Index: {match_index}")

    # Download video
    print("\n" + "=" * 60)
    print("DOWNLOADING VIDEO")
    print("=" * 60)

    video_filename = f"{name}.mp4"
    download_video(url, video_filename)
    print_cwd_size()

    # Load homography matrix
    H_df = pd.read_csv(root / 'homography.csv')
    str_H = H_df.iloc[match_index]['homography_matrix']
    list_of_lists = ast.literal_eval(str_H)
    H = np.array(list_of_lists)

    # Get court corners (for ROI if needed)
    upleft = (H_df.iloc[match_index]['upleft_x'], H_df.iloc[match_index]['upleft_y'])
    upright = (H_df.iloc[match_index]['upright_x'], H_df.iloc[match_index]['upright_y'])
    downleft = (H_df.iloc[match_index]['downleft_x'], H_df.iloc[match_index]['downleft_y'])
    downright = (H_df.iloc[match_index]['downright_x'], H_df.iloc[match_index]['downright_y'])
    court = [upleft, upright, downleft, downright]

    # Load set CSVs
    print("\n" + "=" * 60)
    print("LOADING SET DATA")
    print("=" * 60)

    main_dir = root / name
    sets_data = []
    set_names = []

    for set_num in [1, 2, 3]:
        set_file = main_dir / f'set{set_num}.csv'
        if set_file.exists():
            df = pd.read_csv(set_file)
            df['main_label'] = None
            sets_data.append(df)
            set_names.append(f'set{set_num}')
            print(f"✓ Loaded set{set_num}.csv: {len(df)} strokes")
        else:
            print(f"  set{set_num}.csv not found, skipping")

    if len(sets_data) == 0:
        print("Error: No set data found")
        sys.exit(1)

    # Extract frames for all strokes
    print("\n" + "=" * 60)
    print("EXTRACTING FRAMES")
    print("=" * 60)

    for set_name, df in zip(set_names, sets_data):
        print(f"\nExtracting frames for {set_name}...")

        for i in range(len(df)):
            stroke_dir, main_label = extract_stroke_frames(
                video_path=video_filename,
                df=df,
                index=i,
                match_name=name,
                set_name=set_name,
                output_base_dir=output_base_dir
            )

            if stroke_dir is None:
                print(f"  ✗ Failed to extract {set_name}/{i}")
                continue

            # Update DataFrame with main_label
            df.at[i, 'main_label'] = main_label

            if (i + 1) % 10 == 0:
                print(f"  Extracted {i + 1}/{len(df)} strokes")

        print(f"✓ Completed {set_name}")
    print_cwd_size()
    video_file = Path(f"{name}.mp4")
    if video_file.exists():
        video_file.unlink()
        print(f"✓ Deleted video: {video_file}")
    # Initialize YOLO pose model
    print("\n" + "=" * 60)
    print("INITIALIZING POSE MODEL")
    print("=" * 60)

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    pose_model = YOLO("yolov8n-pose.pt")
    print("✓ YOLOv8 pose model loaded")

    # Process all strokes with pose estimation
    all_poses, all_positions, all_labels, skip_count = process_strokes_to_pkl(
        match_name=name,
        set_dirs=set_names,
        dfs=sets_data,
        pose_model=pose_model,
        H=H,
        device=device,
        output_base_dir=output_base_dir,
        visualize=visualize
    )

    # Save to pkl files
    print("\n" + "=" * 60)
    print("SAVING PKL FILES")
    print("=" * 60)

    poses_file = f"poses_{match_index}.pkl"
    positions_file = f"positions_{match_index}.pkl"
    labels_file = f"labels_{match_index}.pkl"

    with open(poses_file, "wb") as f:
        pickle.dump(all_poses, f)
    print(f"✓ Saved {poses_file}")

    with open(positions_file, "wb") as f:
        pickle.dump(all_positions, f)
    print(f"✓ Saved {positions_file}")

    with open(labels_file, "wb") as f:
        pickle.dump(all_labels, f)
    print(f"✓ Saved {labels_file}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Match: {name}")
    print(f"Total strokes processed: {len(all_poses)}")
    print(f"Strokes skipped: {skip_count}")
    print(
        f"Success rate: {len(all_poses)}/{len(all_poses) + skip_count} ({100 * len(all_poses) / (len(all_poses) + skip_count):.1f}%)")
    print(f"\nOutput files:")
    print(f"  - {poses_file} ({len(all_poses)} strokes, each (30,2,17,3))")
    print(f"  - {positions_file} ({len(all_positions)} strokes, each (2,30,2))")
    print(f"  - {labels_file} ({len(all_labels)} labels)")
    print("=" * 60)
    print(sum(f.stat().st_size for f in Path('.').rglob('*') if f.is_file()) / 1024 ** 3, "GB")
    cleanup_generated_files(name, keep_pkl=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Badminton Pose Extraction Pipeline - Extract poses and positions from match videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python badminton_pose_pipeline.py -m 21
  python badminton_pose_pipeline.py --match_index 21 --visualize
  python badminton_pose_pipeline.py -m 21 --skip-install
        """
    )

    parser.add_argument(
        '-m', '--match_index',
        type=int,
        required=True,
        help='Index of the match in match.csv (e.g., 21)'
    )



    args = parser.parse_args()

    try:
        main(args.match_index)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)