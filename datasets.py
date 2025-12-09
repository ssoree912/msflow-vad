import os
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

__all__ = ('MVTecDataset', 'VisADataset', 'ShanghaiTechDataset', 'RailDataset', 'TxtRailDataset')

MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

class MVTecDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.input_size = c.input_size
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.input_size, InterpolationMode.NEAREST),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        #x = Image.open(x).convert('RGB')
        x = Image.open(x)
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)
            
            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        #
        x = self.normalize(self.transform_x(x))
        #
        if y == 0:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)

VISA_CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 
                    'fryum', 'macaroni1', 'macaroni2', 
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

class VisADataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in VISA_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.input_size = c.input_size
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.input_size, InterpolationMode.NEAREST),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x)
        x = self.normalize(self.transform_x(x))
        if y == 0:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)


class ShanghaiTechDataset(Dataset):
    """Dataset loader for ShanghaiTech-style frame data.

    Expected layout under c.data_path:
    - training/frames/<video_id>/*.png|jpg  (all normal)
    - testing/frames/<video_id>/*.png|jpg
    - testing/test_frame_mask/<video_id>.npy  (frame-level 0/1 labels)
    - testing/test_pixel_mask/<video_id>/*.png (pixel masks, optional)
    """
    def __init__(self, c, is_train=True):
        self.dataset_path = c.data_path
        self.is_train = is_train
        self.input_size = c.input_size
        self.transform_x = T.Compose([
            T.Resize(c.input_size, InterpolationMode.LANCZOS),
            T.ToTensor()])
        self.transform_mask = T.Compose([
            T.Resize(c.input_size, InterpolationMode.NEAREST),
            T.ToTensor()])
        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

        self.x, self.y, self.mask = self.load_dataset_folder()

    def __getitem__(self, idx):
        x_path, y, mask_path = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x_path).convert('RGB')
        x = self.normalize(self.transform_x(x))

        if mask_path is None or y == 0:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask_img = Image.open(mask_path)
            mask = self.transform_mask(mask_img)
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        if self.is_train:
            frame_root = os.path.join(self.dataset_path, 'training', 'frames')
            video_dirs = [d for d in sorted(os.listdir(frame_root))
                          if os.path.isdir(os.path.join(frame_root, d))]
            x, y, mask = [], [], []
            for vid in video_dirs:
                vid_dir = os.path.join(frame_root, vid)
                frame_paths = self._sorted_frames(vid_dir)
                x.extend(frame_paths)
                y.extend([0] * len(frame_paths))
                mask.extend([None] * len(frame_paths))
            return x, y, mask

        # test split
        frame_root = os.path.join(self.dataset_path, 'testing', 'frames')
        frame_mask_root = os.path.join(self.dataset_path, 'testing', 'test_frame_mask')
        pixel_mask_root = os.path.join(self.dataset_path, 'testing', 'test_pixel_mask')

        video_dirs = [d for d in sorted(os.listdir(frame_root))
                      if os.path.isdir(os.path.join(frame_root, d))]
        x, y, mask = [], [], []
        for vid in video_dirs:
            vid_dir = os.path.join(frame_root, vid)
            frame_paths = self._sorted_frames(vid_dir)
            label_path = os.path.join(frame_mask_root, f'{vid}.npy')
            frame_labels = np.load(label_path)
            if len(frame_labels) != len(frame_paths):
                raise ValueError(f'Frame count/label mismatch for {vid}: {len(frame_paths)} frames vs {len(frame_labels)} labels')

            pixel_mask_dir = os.path.join(pixel_mask_root, vid)
            pixel_mask_map = {}
            if os.path.isdir(pixel_mask_dir):
                for fname in os.listdir(pixel_mask_dir):
                    if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                        key = os.path.splitext(fname)[0]
                        pixel_mask_map[key] = os.path.join(pixel_mask_dir, fname)

            for frame_path, label in zip(frame_paths, frame_labels):
                base = os.path.splitext(os.path.basename(frame_path))[0]
                mask_path = pixel_mask_map.get(base)
                x.append(frame_path)
                y.append(int(label))
                mask.append(mask_path)

        assert len(x) == len(y), 'number of x and y should be same'
        return x, y, mask

    def _sorted_frames(self, video_dir):
        frames = [os.path.join(video_dir, f) for f in os.listdir(video_dir)
                  if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
        return sorted(frames)

class RailDataset(Dataset):
    """Dataset loader for rail UVAD frames.

    Expected layout under c.data_path:
      normal_frames/*.jpg|png
      anomaly_frames/*.jpg|png
      anomaly_masks/*.jpg|png   (same basename as anomaly frame)
    Train: only normal_frames.
    Test: normal_frames + anomaly_frames (anomaly uses masks if present).
    """
    def __init__(self, c, is_train=True):
        self.dataset_path = c.data_path
        self.is_train = is_train
        self.input_size = c.input_size
        self.transform_x = T.Compose([
            T.Resize(c.input_size, InterpolationMode.LANCZOS),
            T.ToTensor()])
        self.transform_mask = T.Compose([
            T.Resize(c.input_size, InterpolationMode.NEAREST),
            T.ToTensor()])
        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

        self.x, self.y, self.mask = self.load_dataset_folder()

    def __getitem__(self, idx):
        img_path, label, mask_path = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(img_path).convert('RGB')
        x = self.normalize(self.transform_x(x))
        if label == 0 or mask_path is None:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask_img = Image.open(mask_path)
            mask = self.transform_mask(mask_img)
        return x, label, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        normal_dir = os.path.join(self.dataset_path, 'normal_frames')
        anomaly_dir = os.path.join(self.dataset_path, 'anomaly_frames')
        mask_dir = os.path.join(self.dataset_path, 'anomaly_masks')
        x, y, mask = [], [], []

        def list_images(root):
            return sorted([os.path.join(root, f) for f in os.listdir(root)
                           if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS])

        if self.is_train:
            normal_imgs = list_images(normal_dir)
            x.extend(normal_imgs)
            y.extend([0] * len(normal_imgs))
            mask.extend([None] * len(normal_imgs))
            return x, y, mask

        normal_imgs = list_images(normal_dir)
        x.extend(normal_imgs)
        y.extend([0] * len(normal_imgs))
        mask.extend([None] * len(normal_imgs))

        anomaly_imgs = list_images(anomaly_dir)
        for img_path in anomaly_imgs:
            base = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = None
            for ext in IMAGE_EXTENSIONS:
                cand = os.path.join(mask_dir, base + ext)
                if os.path.exists(cand):
                    mask_path = cand
                    break
            x.append(img_path)
            y.append(1)
            mask.append(mask_path)

        return x, y, mask


class TxtRailDataset(Dataset):
    """Dataset loader that reads samples from train/test txt files.

    Each line: img_path label [mask_path]
    Paths are relative to c.data_path.
    """
    def __init__(self, c, is_train=True):
        self.dataset_path = c.data_path
        self.is_train = is_train
        list_path = c.train_list if is_train else c.test_list
        self.samples = self._read_list(list_path)
        self.input_size = c.input_size
        self.transform_x = T.Compose([
            T.Resize(c.input_size, InterpolationMode.LANCZOS),
            T.ToTensor()])
        self.transform_mask = T.Compose([
            T.Resize(c.input_size, InterpolationMode.NEAREST),
            T.ToTensor()])
        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

    def _read_list(self, list_path):
        samples = []
        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                img_rel = parts[0]
                label = int(parts[1])
                mask_rel = parts[2] if len(parts) > 2 else None
                img_path = os.path.join(self.dataset_path, img_rel)
                mask_path = os.path.join(self.dataset_path, mask_rel) if mask_rel else None
                samples.append((img_path, label, mask_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.samples[idx]
        x = Image.open(img_path).convert('RGB')
        x = self.normalize(self.transform_x(x))
        if label == 0 or not mask_path:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask_img = Image.open(mask_path)
            mask = self.transform_mask(mask_img)
        return x, label, mask
