import cv2
from datetime import datetime
from ultralytics.utils.plotting import Annotator, colors
from model import load_model

def run_segmentation(input_video, output_dir):
    model = load_model()
    names = model.model.names
    cap = cv2.VideoCapture(input_video)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # out file timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_video = f"{output_dir}/segmentation-out-{timestamp}.mp4"

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

    while True:
        ret, im0 = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        results = model.predict(im0)
        annotator = Annotator(im0, line_width=2)

        if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            masks = results[0].masks.xy
            for mask, cls in zip(masks, clss):
                # change made here -> April 14, 2024 14:45
                 if len(mask) > 0:
                    annotator.seg_bbox(mask=mask,
                                    mask_color=colors(int(cls), True),
                                    det_label=names[int(cls)])

        out.write(im0)
        cv2.imshow("segmentation-out", im0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()

    return output_video
