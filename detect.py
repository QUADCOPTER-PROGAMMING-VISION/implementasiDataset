import argparse
import os
import platform
import sys
import time
from pathlib import Path
import torch
import cv2
from models.common import DetectMultiBackend
from utils.general import (
    LOGGER,
    check_img_size,
    check_imshow,
    non_max_suppression,
    scale_boxes,
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

@smart_inference_mode()
def run(
    weights="yolov5n.pt",
    source=0,
    imgsz=(640, 640),
    conf_thres=0.8,
    iou_thres=0.45,
    max_det=10,
    device="",
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
):
    source = str(source)
    webcam = source.isnumeric()
    device = select_device(device)

    if platform.system() == "Windows":
        import pathlib
        pathlib.PosixPath = pathlib.WindowsPath

    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Open video source
    cap = cv2.VideoCapture(int(source))
    if not cap.isOpened():
        print(f"Error: can't open video source{source}")
        return  

    model.warmup(imgsz=(1, 3, *imgsz))

    frame_count = 0
    fps_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: can't read frame from camera!")
            continue  
        frame = cv2.flip(frame, 1)
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255.0
        im = im.permute(2, 0, 1).unsqueeze(0)

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

        annotator = Annotator(frame, line_width=line_thickness, example=str(names))

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    c = int(cls)
                    label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                    annotator.box_label(xyxy, label, color=colors(c, True))

        frame_count += 1
        fps_end_time = time.time()
        fps = frame_count / (fps_end_time - fps_start_time)
        
        h, w, _ = frame.shape  
        fps_text = f"FPS: {fps:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        x = w - text_width - 10  
        y = h - text_height - 10  

        cv2.putText(frame, fps_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

        if cv2.getWindowProperty("Object Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "ahkir.pt", help="model path")
    parser.add_argument("--source", type=str, default="0", help="webcam source")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640], help="inference size")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=10, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness")
    parser.add_argument("--hide-labels", action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision")
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)