"""Utility to inspect ad banner recommendations and visualize dominant heading."""

import argparse
import csv
import math
import os
import random
import sys
from collections import Counter

import cv2

DEFAULT_OUTPUT = 'dominant_heading.png'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='ad_banner_recommendations.csv',
                        help='Path to ad_banner_recommendations.csv')
    parser.add_argument('--video', default='../dataset/videos/Airport-4.mp4',
                        help='Path to the source video to sample a frame from.')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help='Where to save the annotated image.')
    parser.add_argument('--angle-bin', type=float, default=15.0,
                        help='Bin size in degrees when tallying the dominant heading.')
    return parser.parse_args()


def read_angles(csv_path):
    angles = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row.get('dominant_angle_deg')
            if value is None or value == '' or value.lower() == 'none':
                continue
            try:
                angles.append(float(value))
            except ValueError:
                continue
    return angles


def compute_mode_angle(angles, bin_size):
    if not angles:
        return None
    counter = Counter()
    for angle in angles:
        bucket = round(angle / bin_size) * bin_size
        counter[bucket] += 1
    dominant = counter.most_common(1)[0][0]
    return dominant


def sample_frame(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Video not found: {video_path}')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Unable to open video source')
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    target_idx = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f'Failed to read frame {target_idx}')
    cap.release()
    return frame, target_idx


def draw_heading_arrow(frame, angle_deg):
    if frame is None or angle_deg is None:
        return frame
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    length = int(min(w, h) * 0.25)
    radians = math.radians(angle_deg)
    end_x = int(center[0] + length * math.cos(radians))
    end_y = int(center[1] - length * math.sin(radians))
    cv2.arrowedLine(frame, center, (end_x, end_y), (0, 0, 255), 4, tipLength=0.15)
    cv2.putText(frame, f'Dominant heading: {angle_deg:.0f} deg',
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()
    angles = read_angles(args.csv)
    if not angles:
        print('No valid angles found in CSV.', file=sys.stderr)
        sys.exit(1)
    dominant_angle = compute_mode_angle(angles, args.angle_bin)
    frame, frame_idx = sample_frame(args.video)
    draw_heading_arrow(frame, dominant_angle)
    cv2.putText(frame, f'Frame {frame_idx}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(args.output, frame)
    print(f'Saved visualization to {args.output}. Dominant heading: {dominant_angle:.2f} deg')


if __name__ == '__main__':
    main()
