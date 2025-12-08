import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm

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
        name_node = obj.find("name")
        name = (name_node.text or "").strip().lower() if name_node is not None else ""
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        # 일부 객체는 name 태그가 비어 있음 → bbox가 있으면 이상으로 취급
        if name == "person" or name == "":
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
    if not folder.endswith("_auto_annots"):
        continue

    xml_dir = os.path.join(ANN_ROOT, folder)
    video_name = folder.replace("_auto_annots", "")
    # 프레임 폴더만 사용
    frames_dir = os.path.join(FRAMES_ROOT, video_name)
    if not os.path.isdir(frames_dir):
        print(f"[WARN] frames dir not found for {video_name}: {frames_dir}")
        continue

    xml_files = sorted([f for f in os.listdir(xml_dir) if f.endswith(".xml")])

    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        f_idx = frame_index_from_xml(xml_file)
        stem = os.path.splitext(xml_file)[0]
        img_path = os.path.join(frames_dir, stem + ".jpg")
        if not os.path.exists(img_path):
            print(f"[WARN] frame file not found {img_path}")
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
