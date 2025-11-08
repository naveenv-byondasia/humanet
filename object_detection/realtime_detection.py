import csv
import math
import numpy as np
import os
import sys
import time
import tensorflow.compat.v1 as tf
import cv2

from collections import defaultdict, OrderedDict, deque

sys.path.insert(0, os.path.abspath(".."))
from utils import label_map_util

# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'
MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'

PATH_TO_CKPT = os.path.join(MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

PATH_TO_VIDEO = '../dataset/videos'

# Performance/visualization tuning.
TARGET_FRAME_WIDTH = 720  # Resize frames before detection; set to None to keep original size.
DETECTION_INTERVAL = 7    # Run detector every N frames (increase to trade accuracy for FPS).
MIN_SCORE_THRESH = 0.45
PERSON_CLASS_ID = 1
MAX_TRACKER_DISTANCE = 90
MAX_TRACKER_DISAPPEARED = 30
MAX_TRAIL_LENGTH = 40
AD_GRID_ROWS = 3
AD_GRID_COLS = 3
AD_ALPHA = 0.3
RECOMMENDATION_FILE = 'ad_banner_recommendations.csv'
ANGLE_BIN_DEGREES = 15

tf.disable_v2_behavior()

print ('loading model..')

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')
	
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


class CentroidTracker:
	"""Lightweight object tracker that matches detections via centroid distance."""
	def __init__(self, max_disappeared=MAX_TRACKER_DISAPPEARED, max_distance=MAX_TRACKER_DISTANCE):
		self.next_object_id = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.max_disappeared = max_disappeared
		self.max_distance = max_distance

	def register(self, centroid):
		self.objects[self.next_object_id] = centroid
		self.disappeared[self.next_object_id] = 0
		self.next_object_id += 1

	def deregister(self, object_id):
		self.objects.pop(object_id, None)
		self.disappeared.pop(object_id, None)

	def update(self, rects):
		if len(rects) == 0:
			for object_id in list(self.disappeared.keys()):
				self.disappeared[object_id] += 1
				if self.disappeared[object_id] > self.max_disappeared:
					self.deregister(object_id)
			return dict(self.objects)

		input_centroids = np.zeros((len(rects), 2), dtype="int")
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			input_centroids[i] = (cX, cY)

		if len(self.objects) == 0:
			for i in range(len(input_centroids)):
				self.register(input_centroids[i])
			return dict(self.objects)

		object_ids = list(self.objects.keys())
		object_centroids = np.array(list(self.objects.values()))
		distances = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2)
		rows = distances.min(axis=1).argsort()
		cols = distances.argmin(axis=1)[rows]

		used_rows = set()
		used_cols = set()

		for (row, col) in zip(rows, cols):
			if row in used_rows or col in used_cols:
				continue
			if distances[row, col] > self.max_distance:
				continue
			object_id = object_ids[row]
			self.objects[object_id] = input_centroids[col]
			self.disappeared[object_id] = 0
			used_rows.add(row)
			used_cols.add(col)

		unused_rows = set(range(distances.shape[0])).difference(used_rows)
		unused_cols = set(range(distances.shape[1])).difference(used_cols)

		if distances.shape[0] >= distances.shape[1]:
			for row in unused_rows:
				object_id = object_ids[row]
				self.disappeared[object_id] += 1
				if self.disappeared[object_id] > self.max_disappeared:
					self.deregister(object_id)
		else:
			for col in unused_cols:
				self.register(input_centroids[col])

		return dict(self.objects)

	def current_objects(self):
		return dict(self.objects)


class TrajectoryManager:
	"""Maintains short trails for tracked objects."""
	def __init__(self, max_length=MAX_TRAIL_LENGTH):
		self.max_length = max_length
		self.tracks = defaultdict(lambda: deque(maxlen=self.max_length))

	def update(self, objects):
		active_ids = set(objects.keys())
		for object_id in list(self.tracks.keys()):
			if object_id not in active_ids:
				self.tracks.pop(object_id, None)

		for object_id, centroid in objects.items():
			self.tracks[object_id].append(tuple(centroid))

	def draw(self, frame):
		for object_id, points in self.tracks.items():
			if len(points) < 2:
				continue
			for i in range(1, len(points)):
				pt1 = points[i - 1]
				pt2 = points[i]
				thickness = int(max(1, 4 - (i * 0.3)))
				cv2.line(frame, pt1, pt2, (0, 255, 255), thickness)
			cv2.putText(frame, f'ID {object_id}', points[-1], cv2.FONT_HERSHEY_SIMPLEX,
					0.5, (255, 255, 255), 1, cv2.LINE_AA)

	def get_tracks(self):
		return dict(self.tracks)


def maybe_resize(frame, target_width=None):
	if not target_width:
		return frame
	height, width = frame.shape[:2]
	if width <= target_width:
		return frame
	scale = target_width / float(width)
	new_size = (target_width, int(height * scale))
	return cv2.resize(frame, new_size)


def extract_detections(boxes, scores, classes, num_detections, frame_shape):
	im_height, im_width = frame_shape[:2]
	detections = []
	for i in range(int(num_detections[0])):
		score = scores[0][i]
		if score < MIN_SCORE_THRESH:
			continue
		class_id = int(classes[0][i])
		if class_id != PERSON_CLASS_ID:
			continue
		ymin, xmin, ymax, xmax = boxes[0][i]
		left = int(xmin * im_width)
		right = int(xmax * im_width)
		top = int(ymin * im_height)
		bottom = int(ymax * im_height)
		detections.append((left, top, right, bottom, class_id, float(score)))
	return detections


def draw_detections(frame, detections, tracked_objects, category_index):
	used_ids = set()
	for (left, top, right, bottom, class_id, score) in detections:
		color = (0, 255, 0)
		cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
		label = category_index.get(class_id, {'name': str(class_id)})['name']
		object_id = _match_object_id((left, top, right, bottom), tracked_objects, used_ids)
		display = f"{label}: {score:.2f}"
		if object_id is not None:
			display += f" | ID {object_id}"
		cv2.putText(frame, display, (left, max(15, top - 10)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def _match_object_id(box, tracked_objects, used_ids):
	left, top, right, bottom = box
	for object_id, (cX, cY) in tracked_objects.items():
		if object_id in used_ids:
			continue
		if left <= cX <= right and top <= cY <= bottom:
			used_ids.add(object_id)
			return object_id
	return None


def compute_heading_angles(trajectory_manager):
	angles = {}
	for object_id, points in trajectory_manager.get_tracks().items():
		if len(points) < 2:
			continue
		x1, y1 = points[-2]
		x2, y2 = points[-1]
		dx = x2 - x1
		dy = y2 - y1
		if dx == 0 and dy == 0:
			continue
		angle = math.degrees(math.atan2(-(dy), dx))  # invert y because image origin is top-left
		angles[object_id] = angle
	return angles


def dominant_angle(angles):
	if not angles:
		return None
	angle_values = list(angles.values())
	bins = np.arange(-180, 180 + ANGLE_BIN_DEGREES, ANGLE_BIN_DEGREES)
	hist, edges = np.histogram(angle_values, bins=bins)
	if hist.sum() == 0:
		return None
	max_idx = hist.argmax()
	return (edges[max_idx] + edges[max_idx + 1]) / 2.0


def recommend_ad_region(frame_shape, detections):
	if frame_shape is None:
		return None, None, None
	height, width = frame_shape[:2]
	cell_h = height / AD_GRID_ROWS
	cell_w = width / AD_GRID_COLS
	heatmap = np.zeros((AD_GRID_ROWS, AD_GRID_COLS), dtype=np.float32)
	for (left, top, right, bottom, _, _) in detections:
		cX = max(0, min(width - 1, int((left + right) / 2)))
		cY = max(0, min(height - 1, int((top + bottom) / 2)))
		col = min(AD_GRID_COLS - 1, int(cX / cell_w))
		row = min(AD_GRID_ROWS - 1, int(cY / cell_h))
		area = max(1, (right - left) * (bottom - top))
		heatmap[row, col] += area

	row = 0
	col = 0
	if np.allclose(heatmap, 0):
		row, col = AD_GRID_ROWS - 1, AD_GRID_COLS - 1
	else:
		row, col = np.unravel_index(np.argmin(heatmap), heatmap.shape)
	x1 = int(col * cell_w)
	y1 = int(row * cell_h)
	x2 = int(min(width, (col + 1) * cell_w))
	y2 = int(min(height, (row + 1) * cell_h))
	return (x1, y1, x2, y2), (row, col), heatmap


def draw_ad_region(frame, region, text):
	if region is None:
		return
	x1, y1, x2, y2 = region
	overlay = frame.copy()
	cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
	cv2.addWeighted(overlay, AD_ALPHA, frame, 1 - AD_ALPHA, 0, dst=frame)
	cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)
	if text:
		cv2.putText(frame, text, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX,
			0.55, (255, 255, 255), 2, cv2.LINE_AA)


def log_recommendation(logs, frame_idx, timestamp, grid_cell, region, angle):
	row, col = grid_cell if grid_cell else (None, None)
	entry = {
		'frame': frame_idx,
		'timestamp': timestamp,
		'grid_row': row,
		'grid_col': col,
		'region': region,
		'dominant_angle_deg': angle
	}
	logs.append(entry)


def persist_recommendations(logs, path):
	if not logs:
		return
	fieldnames = ['frame', 'timestamp', 'grid_row', 'grid_col', 'region', 'dominant_angle_deg']
	with open(path, 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for row in logs:
			writer.writerow(row)


with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		cap = cv2.VideoCapture(os.path.join(PATH_TO_VIDEO, 'Airport-4.mp4'))
		if not cap.isOpened():
			raise RuntimeError('Unable to open video source.')

		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
		boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
		scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
		classes_tensor = detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections_tensor = detection_graph.get_tensor_by_name('num_detections:0')

		tracker = CentroidTracker()
		trajectory_manager = TrajectoryManager()
		frame_index = 0
		fps = 0.0
		prev_time = time.time()
		last_detections = []
		dominant_heading = None
		recommendation_log = []

		while True:
			ret, frame = cap.read()
			if not ret:
				break

			frame = maybe_resize(frame, TARGET_FRAME_WIDTH)
			frame_index += 1
			should_detect = (frame_index % DETECTION_INTERVAL) == 0
			display_frame = frame.copy()

			if should_detect:
				image_np = np.array(frame)
				if image_np.size == 0:
					continue
				image_np_expanded = np.expand_dims(image_np, axis=0)
				(boxes, scores, classes, num_detections) = sess.run(
					[boxes_tensor, scores_tensor, classes_tensor, num_detections_tensor],
					feed_dict={image_tensor: image_np_expanded})

				last_detections = extract_detections(boxes, scores, classes, num_detections, frame.shape)
				rects = [(x1, y1, x2, y2) for (x1, y1, x2, y2, _, _) in last_detections]
				tracked_objects = tracker.update(rects)
				trajectory_manager.update(tracked_objects)
				current_time = time.time()
				fps = 1.0 / max(1e-6, (current_time - prev_time))
				prev_time = current_time
				timestamp = time.time()
			else:
				tracked_objects = tracker.current_objects()

			angles = compute_heading_angles(trajectory_manager)
			angle_value = dominant_angle(angles)
			if angle_value is not None:
				dominant_heading = angle_value
			ad_region, grid_cell, heatmap = recommend_ad_region(display_frame.shape, last_detections)
			if should_detect:
				log_recommendation(recommendation_log, frame_index, timestamp, grid_cell, ad_region, dominant_heading)

			draw_detections(display_frame, last_detections, tracked_objects, category_index)
			trajectory_manager.draw(display_frame)
			heading_text = 'Heading: --'
			if dominant_heading is not None:
				heading_text = f'Heading: {dominant_heading:.0f} deg'
			cv2.putText(display_frame, heading_text, (10, 75),
				cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
			if ad_region:
				grid_text = f'Ad zone r{grid_cell[0]}c{grid_cell[1]}' if grid_cell else 'Ad zone'
				if dominant_heading is not None:
					grid_text += f' | {dominant_heading:.0f} deg'
				draw_ad_region(display_frame, ad_region, grid_text)
			cv2.putText(display_frame, f'FPS: {fps:.1f}', (10, 25),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(display_frame, f'Detect every {DETECTION_INTERVAL} frame(s)', (10, 50),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

			cv2.imshow('frame', display_frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()
		persist_recommendations(recommendation_log, RECOMMENDATION_FILE)
