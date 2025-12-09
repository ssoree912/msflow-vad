import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

# 경로 설정 (필요에 맞게 수정)
ANN_ROOT = "./data/rail/annots/2024-04-25_TUEV_Elchingen"  # XML이 들어있는 *_auto_annots 상위 경로
FRAMES_ROOT = "./data/rail/imgs/2024-04-25_TUEV_Elchingen"  # 추출된 프레임이 디렉터리로 있는 상위 경로
SAVE_ROOT = "./data/rail/rail_uvad_dataset"

os.makedirs(SAVE_ROOT, exist_ok=True)
NORMAL_DIR = os.path.join(SAVE_ROOT, "normal_frames")
ANOMALY_DIR = os.path.join(SAVE_ROOT, "anomaly_frames")
MASK_DIR = os.path.join(SAVE_ROOT, "anomaly_masks")
for d in (NORMAL_DIR, ANOMALY_DIR, MASK_DIR):
    os.makedirs(d, exist_ok=True)


def parse_person_boxes(xml_path):
    boxes = []
    root = ET.parse(xml_path).getroot()
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        # 라벨이 비어 있거나 다른 클래스여도 bbox가 있으면 이상으로 취급
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def frame_index_from_xml(xml_filename):
    stem = os.path.splitext(xml_filename)[0]
    last_part = stem.split("_")[-1]
    try:
        return int(last_part)
    except ValueError:
        return None


for folder in tqdm(sorted(os.listdir(ANN_ROOT))):
    xml_dir = os.path.join(ANN_ROOT, folder)
    if not os.path.isdir(xml_dir):
        continue
    xml_files = sorted([f for f in os.listdir(xml_dir) if f.endswith(".xml")])
    if not xml_files:
        continue

    # 프레임 폴더만 사용
    video_name = folder.replace("_auto_annots", "")
    candidate_dirs = [
        os.path.join(FRAMES_ROOT, video_name),
        os.path.join(FRAMES_ROOT, os.path.splitext(video_name)[0]),
        os.path.join(FRAMES_ROOT, video_name.replace(".mp4", "")),
    ]
    frames_dir = None
    for cand in candidate_dirs:
        if os.path.isdir(cand):
            frames_dir = cand
            break
    if frames_dir is None:
        print(f"[WARN] frames dir not found for {video_name}: tried {candidate_dirs}")
        continue

    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        f_idx = frame_index_from_xml(xml_file)
        stem = os.path.splitext(xml_file)[0]
        img_path = None
        for ext in IMAGE_EXTS:
            cand = os.path.join(frames_dir, stem + ext)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            print(f"[WARN] frame file not found for {xml_file} in {frames_dir}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] failed to read image {img_path}")
            continue

        boxes = parse_person_boxes(xml_path)
        frame_name = video_name.replace(".mp4", f"_{f_idx:06d}.jpg")
        if len(boxes) == 0:
            cv2.imwrite(os.path.join(NORMAL_DIR, frame_name), img)
        else:
            cv2.imwrite(os.path.join(ANOMALY_DIR, frame_name), img)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for (x1, y1, x2, y2) in boxes:
                mask[y1:y2, x1:x2] = 255
            cv2.imwrite(os.path.join(MASK_DIR, frame_name), mask)

print("done.")
