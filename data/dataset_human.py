# This code is designed to work within the LVSM framework structure.

import random
import traceback
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F


class HumanGSDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Determine mode
        self.inference = self.config.inference.get("if_inference", False)

        # Select the correct path based on mode
        if self.inference:
            json_path = self.config.training.get("test_dataset_path")
            print(f"Loading TEST dataset from: {json_path}")
        else:
            json_path = self.config.training.get("train_dataset_path")
            print(f"Loading TRAIN dataset from: {json_path}")

        if not json_path:
            raise ValueError("Dataset path not found in config! Set 'train_dataset_path' and 'test_dataset_path'.")

        try:
            with open(json_path, 'r') as f:
                self.all_scene_paths = json.load(f)
        except Exception as e:
            print(f"Error reading dataset paths from '{json_path}'")
            raise e

        if len(self.all_scene_paths) == 0:
            raise ValueError(f"Dataset at {json_path} is EMPTY! Check preprocessing.")

        if self.inference:
            self.view_idx_list = dict()
            if self.config.inference.get("view_idx_file_path", None) is not None:
                if os.path.exists(self.config.inference.view_idx_file_path):
                    with open(self.config.inference.view_idx_file_path, 'r') as f:
                        self.view_idx_list = json.load(f)
                        self.view_idx_list_filtered = [k for k, v in self.view_idx_list.items() if v is not None]

                    # Filter scenes based on scene_name for inference
                    filtered_scene_paths = []
                    for scene_meta in self.all_scene_paths:
                        scene_name = scene_meta["scene_name"]
                        if scene_name in self.view_idx_list_filtered:
                            filtered_scene_paths.append(scene_meta)
                    self.all_scene_paths = filtered_scene_paths

    def __len__(self):
        if self.inference:
            return len(self.all_scene_paths)
        return len(self.all_scene_paths) * 1000

    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        # HumanRAM uses 512x512 images (or 256x256 for low-res training)
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            try:
                image = Image.open(cur_image_path)
            except Exception as e:
                print(f"Error loading image {cur_image_path}: {e}")
                image = Image.new('RGB', (resize_h, resize_h))

            original_image_w, original_image_h = image.size

            # Aspect-ratio preserving resize
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)

            image = image.resize((resize_w, resize_h), resample=Image.LANCZOS)
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            # Adjust intrinsics for resize/crop
            fxfycxcy = np.array(cur_frame["fxfycxcy"])
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()

            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)

        w2cs = np.stack([np.array(frame["w2c"]) for frame in frames_chosen])
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws

    def view_selector(self, frames):
        total = len(frames)
        num_views = self.config.training.num_views

        # Handle short clips by duplication
        if total < num_views:
            indices = list(range(total))
            while len(indices) < num_views: indices.append(indices[-1])
            return indices[:num_views]

        view_selector_config = self.config.training.view_selector
        min_frame_dist = view_selector_config.get("min_frame_dist", 25)
        max_frame_dist = min(total - 1, view_selector_config.get("max_frame_dist", 100))

        # Fallback if distance constraints cannot be met
        eff_min = min(min_frame_dist, total - 2)
        if max_frame_dist <= eff_min:
             return sorted(random.sample(range(total), num_views))

        frame_dist = random.randint(eff_min, max_frame_dist)

        # Safety check for start frame
        max_start = total - frame_dist - 1
        if max_start < 0: return sorted(random.sample(range(total), num_views))

        start_frame = random.randint(0, max_start)
        end_frame = start_frame + frame_dist

        # Randomly sample frames between start and end
        needed_samples = num_views - 2
        available_range = list(range(start_frame + 1, end_frame))

        if len(available_range) >= needed_samples:
            sampled_frames = random.sample(available_range, needed_samples)
        else:
            sampled_frames = available_range
            while len(sampled_frames) < needed_samples:
                 sampled_frames.append(random.choice(available_range))

        image_indices = [start_frame, end_frame] + sampled_frames
        return image_indices

    def __getitem__(self, idx):
        attempt_idx = idx % len(self.all_scene_paths)

        for _ in range(20):
        # Try 20 times before failing
            try:
                # Load Metadata for the selected subject
                subject_metadata = self.all_scene_paths[attempt_idx]
                frames = subject_metadata["frames"]
                scene_name = subject_metadata["scene_name"]

                if len(frames) == 0: raise ValueError("Empty frames")

                # Select specific frames (views/timestamps)
                if self.inference and scene_name in self.view_idx_list:
                    current_view_idx = self.view_idx_list[scene_name]
                    image_indices = current_view_idx["context"] + current_view_idx["target"]
                else:
                    image_indices = self.view_selector(frames)
                    if image_indices is None:
                        attempt_idx = random.randint(0, len(self.all_scene_paths) - 1)
                        continue

                if self.inference:
                    json_path = self.config.training.get("test_dataset_path")
                else:
                    json_path = self.config.training.get("train_dataset_path")

                root_dir = os.path.dirname(json_path) 

                image_paths_chosen = [os.path.join(root_dir, frames[ic]["image_path"]) for ic in image_indices]
                frames_chosen = [frames[ic] for ic in image_indices]

                # Fast existence check
                if not os.path.exists(image_paths_chosen[0]):
                    attempt_idx = random.randint(0, len(self.all_scene_paths) - 1)
                    continue

                # Load Images & Cameras
                input_images, input_intrinsics, input_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)

                # Load SMPL-X Data (Dynamic/Generalized)
                posed_vertices_list = []
                canonical_vertices_list = []
                faces_list = []
                transforms_list = []

                for frame_meta in frames_chosen:
                    smpl_rel_path = frame_meta["smpl_path"]
                    smpl_full_path = os.path.join(root_dir, smpl_rel_path)

                    smpl_data = np.load(smpl_full_path)
                    v_posed = torch.from_numpy(smpl_data['v_posed']).float()
                    v_cano = torch.from_numpy(smpl_data['v_cano']).float()
                    faces = torch.from_numpy(smpl_data['faces']).long()
                    transforms = torch.from_numpy(smpl_data['joint_transforms']).float()

                    posed_vertices_list.append(v_posed)
                    canonical_vertices_list.append(v_cano)
                    faces_list.append(faces)
                    transforms_list.append(transforms)

                # Stack: [V, N_verts, 3]
                smpl_posed_vertices = torch.stack(posed_vertices_list, dim=0)
                smpl_canonical_vertices = torch.stack(canonical_vertices_list, dim=0)
                smpl_faces = torch.stack(faces_list, dim=0)
                joint_transforms = torch.stack(transforms_list, dim=0)

                image_indices_tensor = torch.tensor(image_indices).long().unsqueeze(-1)
                scene_indices_tensor = torch.full_like(image_indices_tensor, attempt_idx)
                indices = torch.cat([image_indices_tensor, scene_indices_tensor], dim=-1)

                return {
                    "image": input_images,
                    "c2w": input_c2ws,
                    "fxfycxcy": input_intrinsics,
                    "index": indices,
                    "scene_name": scene_name,
                    "smpl_posed_vertices": smpl_posed_vertices,
                    "smpl_canonical_vertices": smpl_canonical_vertices,
                    "smpl_faces": smpl_faces,
                    "joint_transforms": joint_transforms
                }

            except Exception as e:
                print(f"Error loading {scene_name}: {e}")
                attempt_idx = random.randint(0, len(self.all_scene_paths) - 1)
                continue

        raise RuntimeError("Failed to load valid batch after retries.")