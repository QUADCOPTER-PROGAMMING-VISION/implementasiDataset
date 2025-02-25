import argparse
import os
import platform
import sys
import time
from pathlib import Path
import torch
import cv2
import pygame
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode

# Inisialisasi Pygame untuk tampilan pergerakan drone
pygame.init()
screen = pygame.display.set_mode((400, 200))
pygame.display.set_caption("Drone Movement Console")
font = pygame.font.Font(None, 36)


def show_drone_movement(movement):
    screen.fill((0, 0, 0))
    text = font.render(movement, True, (0, 255, 0))
    screen.blit(text, (50, 80))
    pygame.display.flip()


def determine_movement(absis, ordinat):
    movement = "Drone Diam"
    camera_adjustment = ""

    if absis < 1:
        movement = "Drone ke Kiri"
        camera_adjustment = "Geser Kamera ke Kiri"
    elif absis > 1:
        movement = "Drone ke Kanan"
        camera_adjustment = "Geser Kamera ke Kanan"
    if ordinat < 1:
        movement += " | Drone ke Bawah"
        camera_adjustment += " | Geser Kamera ke Bawah"
    elif ordinat > 1:
        movement += " | Drone ke Atas"
        camera_adjustment += " | Geser Kamera ke Atas"

    print(movement)
    print(camera_adjustment)
    show_drone_movement(movement + "\n" + camera_adjustment)
    time.sleep(1)


@smart_inference_mode()
def run(
        weights="yolov5n.pt",
        source=2,
        imgsz=(640, 640),
        conf_thres=0.5,
        iou_thres=0.45,
        max_det=10,
        device="",
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
):
    source = str(source)
    device = select_device(device)

    if platform.system() == "Windows":
        import pathlib
        pathlib.PosixPath = pathlib.WindowsPath

    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    cap = cv2.VideoCapture(int(source))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"Error: can't open video source {source}")
        return

    model.warmup(imgsz=(1, 3, *imgsz))
    frame_count = 0
    fps_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: can't read frame from camera!")
            continue

        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255.0
        im = im.permute(2, 0, 1).unsqueeze(0)

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

        annotator = Annotator(frame, line_width=line_thickness, example=str(names))

        h, w, _ = frame.shape
        zone_width = w / 3
        zone_height = h / 3

        # Gambar grid di tampilan kamera
        for i in range(1, 3):
            cv2.line(frame, (int(i * zone_width), 0), (int(i * zone_width), h), (255, 255, 255), 1)
            cv2.line(frame, (0, int(i * zone_height)), (w, int(i * zone_height)), (255, 255, 255), 1)

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2

                    absis = min(2, max(0, int(x_center // zone_width)))
                    ordinat = min(2, max(0, int(y_center // zone_height)))

                    c = int(cls)
                    label = f"{names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    print(f"Objek {names[c]} di Grid [{absis}], [{ordinat}]")
                    determine_movement(absis, ordinat)

        cv2.imshow("Object Detection", frame)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                sys.exit()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty("Object Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov5n.pt", help="model path")
    parser.add_argument("--source", type=str, default="0", help="DroidCam source")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640, 640], help="inference size")
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
