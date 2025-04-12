# # processor.py

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
# import os

# def process_video(input_path, output_path, progress_callback=None):
#     model = YOLO('best.pt')
#     tracker = DeepSort(max_age=30)
#     track_histories = {}

#     cap = cv2.VideoCapture(input_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     frame_count = 0
#     max_frames = int(fps * 60)  # Process only first 60 seconds

#     while frame_count < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model.predict(frame, conf=0.3)
#         detections = []

#         for det in results[0].boxes:
#             bbox = det.xyxy[0].cpu().numpy()
#             x1, y1, x2, y2 = bbox
#             w, h = x2 - x1, y2 - y1
#             conf = det.conf.cpu().item()
#             detections.append(([x1, y1, w, h], conf))

#         if detections:
#             tracks = tracker.update_tracks(detections, frame=frame)
#         else:
#             tracks = []

#         for track in tracks:
#             if not track.is_confirmed():
#                 continue

#             track_id = track.track_id
#             ltrb = track.to_ltrb()
#             x1, y1, x2, y2 = map(int, ltrb)
#             center = ((x1 + x2) // 2, (y1 + y2) // 2)

#             if track_id not in track_histories:
#                 track_histories[track_id] = []
#             track_histories[track_id].append(center)

#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             if len(track_histories[track_id]) > 1:
#                 for i in range(1, len(track_histories[track_id])):
#                     cv2.line(frame,
#                              track_histories[track_id][i - 1],
#                              track_histories[track_id][i],
#                              (255, 0, 0), 2)

#         out.write(frame)
#         frame_count += 1
#         if progress_callback:
#             progress_callback(frame_count / total_frames)

#     cap.release()
#     out.release()


import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

model = YOLO('best.pt')
tracker = DeepSort(max_age=30)

def process_with_id_only(input_path, output_folder):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(output_folder, "id_output.mp4")
    # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'H264'), fps, (width, height))
    max_frames = int(fps * 20)
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.3)
        annotated_frame = results[0].plot()

        out.write(annotated_frame)
        frame_count += 1

    cap.release()
    out.release()
    return output_path


def process_with_tracking(input_path, output_folder):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(output_folder, "tracking_output.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'H264'), fps, (width, height))

    max_frames = int(fps * 20)
    frame_count = 0

    track_histories = {}

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.3)

        detections_for_deepsort = []
        for det in results[0].boxes:
            bbox = det.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            conf = det.conf.cpu().item()
            detections_for_deepsort.append(([x1, y1, w, h], conf))

        if detections_for_deepsort:
            tracks = tracker.update_tracks(detections_for_deepsort, frame=frame)
        else:
            tracks = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if track_id not in track_histories:
                track_histories[track_id] = []
            track_histories[track_id].append(center)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if len(track_histories[track_id]) > 1:
                for i in range(1, len(track_histories[track_id])):
                    cv2.line(frame,
                             track_histories[track_id][i - 1],
                             track_histories[track_id][i],
                             (255, 0, 0), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    return output_path
