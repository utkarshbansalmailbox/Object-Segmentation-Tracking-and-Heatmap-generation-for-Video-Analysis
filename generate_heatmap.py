import os
import cv2
from datetime import datetime
from ultralytics import YOLO
from ultralytics.solutions import heatmap
from model import load_model

def get_latest_output_video(output_dir):
    video_files = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]

    # Sort the list by modification time to get the latest video file
    latest_video = sorted(video_files, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)

    if latest_video:
        return os.path.join(output_dir, latest_video[0])
    else:
        print(f"{output_dir} is not a directory.")
        return None

def run_heatmap_generation(input_video, output_dir):
    model = load_model()
    latest_output_video = get_latest_output_video(output_dir)

    if latest_output_video:
        cap = cv2.VideoCapture(latest_output_video)
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # timestamp for out file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_video = os.path.join(output_dir, f"heatmap-output-{timestamp}.mp4")


        video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))


        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_HSV,
                             imw=w,
                             imh=h,
                             view_img=True,
                             shape="circle", # or rect -> rectangle
                             classes_names=model.names)

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            tracks = model.track(im0, persist=True, show=True)

            im0 = heatmap_obj.generate_heatmap(im0, tracks)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
