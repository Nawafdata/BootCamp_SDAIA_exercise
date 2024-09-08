import streamlit as st
import cv2
import torch
import tempfile
import os

# Load YOLOv5 model (you can replace 'yolov5s' with 'yolov5m', 'yolov5l', etc.)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def process_video(video_file):
    # Read video
    vid = cv2.VideoCapture(video_file)

    # Get video information
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video file
    out_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Process each frame
    for _ in range(total_frames):
        ret, frame = vid.read()
        if not ret:
            break

        # YOLO object detection
        results = model(frame)

        # Draw the detection results on the frame
        for _, row in results.pandas().xyxy[0].iterrows():
            x1, y1, x2, y2, conf, cls, label = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['class'], row['name']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write frame to the output video
        out.write(frame)

    # Release resources
    vid.release()
    out.release()

    return out_video_path

# Streamlit UI
st.title('YOLO Object Detection on Video')
st.write('Upload a video file to perform object detection using YOLO.')

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    st.video(uploaded_file)

    st.write("Processing the video... This may take some time.")
    processed_video_path = process_video(tmp_file_path)
    
    st.write("Object detection complete. Download or view the processed video below.")
    st.video(processed_video_path)
    st.download_button("Download Processed Video", data=open(processed_video_path, 'rb'), file_name="processed_video.mp4")

    # Clean up temporary files
    os.remove(tmp_file_path)
    os.remove(processed_video_path)
