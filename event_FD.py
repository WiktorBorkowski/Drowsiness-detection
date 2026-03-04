import os
import threading
import time
import argparse
import cv2
import numpy as np
import onnxruntime as ort

# Disable unnecessary modules
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_YOLO_WORLD_ENABLED"] = "False"

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent


MODEL_PATH = "best3.onnx"
INPUT_SIZE = 320 
INFERENCE_INTERVAL = 0.1   
FACE_CLASS_INDEX = 0       # Face is the first label

latest_frame = None
current_predictions = []
frame_lock = threading.Lock()
stop_flag = False

def load_model():
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4 
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(MODEL_PATH, sess_options=session_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return session, input_name

def ai_worker(session, input_name):
    global latest_frame, current_predictions, stop_flag
    last_infer_time = 0

    while not stop_flag:
        now = time.time()
        if latest_frame is None or (now - last_infer_time < INFERENCE_INTERVAL):
            time.sleep(0.005)
            continue

        with frame_lock:
            frame = latest_frame.copy()

        frame_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))

        if len(frame_resized.shape) == 2:
            img = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2RGB)
        else:
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        try:
            outputs = session.run(None, {input_name: img})
            preds = np.squeeze(outputs[0]).T 
            
            # Extract potential face boxes and their scores
            class_ids = np.argmax(preds[:, 4:], axis=1)
            face_mask = (class_ids == FACE_CLASS_INDEX)
            face_preds = preds[face_mask]
            
            boxes = []
            confidences = []

            for i in range(len(face_preds)):
                conf = face_preds[i, 4 + FACE_CLASS_INDEX]
                if conf > 0.2: # Detection threshold
                    cx, cy, w, h = face_preds[i, :4]
                    # NMS needs [top-left-x, top-left-y, width, height]
                    x = int(cx - w/2)
                    y = int(cy - h/2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(conf))

            # 2. Apply NMS (remove duplicates)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.25)
            
            final_faces = []
            if len(indices) > 0:
                for i in indices.flatten():
                    # Re-package into correct format
                    b = boxes[i]
                    conf = confidences[i]
                    final_faces.append([b[0] + b[2]/2, b[1] + b[3]/2, b[2], b[3], conf])
            
            current_predictions = final_faces
                
        except Exception as e:
            print("Inference Error:", e)

        last_infer_time = now

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-event-file', default="")
    return parser.parse_args()

def main():
    global latest_frame, current_predictions, stop_flag
    args = parse_args()
    session, input_name = load_model()

    mv_iterator = EventsIterator(input_path=args.input_event_file, delta_t=35000)
    height, width = mv_iterator.get_size()

    if not is_live_camera(args.input_event_file):
        mv_iterator = LiveReplayEventsIterator(mv_iterator)

    t = threading.Thread(target=ai_worker, args=(session, input_name), daemon=True)
    t.start()

    with MTWindow(title="Pi5 Face-Only Inference", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        def keyboard_cb(key, scancode, action, mods):
            if key in (UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q):
                window.set_close_flag()
        window.set_keyboard_callback(keyboard_cb)

        event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=width, sensor_height=height,
            fps=15, palette=ColorPalette.Gray 
        )

        scale_x = width / INPUT_SIZE
        scale_y = height / INPUT_SIZE

        def on_cd_frame_cb(ts, cd_frame):
            global latest_frame, current_predictions
            with frame_lock:
                latest_frame = cd_frame 

            if len(cd_frame.shape) == 2:
                display_frame = cv2.cvtColor(cd_frame, cv2.COLOR_GRAY2BGR)
            else:
                display_frame = cd_frame

            for pred in current_predictions:
                # Get Face confidence score
                conf = pred[4]
                
                if conf > 0.1: 
                    cx, cy, w, h = pred[:4]
                    x1 = int((cx - w/2) * scale_x)
                    y1 = int((cy - h/2) * scale_y)
                    x2 = int((cx + w/2) * scale_x)
                    y2 = int((cy + h/2) * scale_y)

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Face: {conf:.2f}", (x1, y1-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            window.show_async(display_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)
            if window.should_close():
                break

    stop_flag = True
    t.join()

if __name__ == "__main__":
    main()
