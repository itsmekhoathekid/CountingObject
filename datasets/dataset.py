import json
import numpy as np
import random
from torchvision import transforms
import torch
import cv2
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
import os
from transformers import CLIPTokenizer # Cần cài đặt transformers

def random_crop(img_h, img_w, crop_h, crop_w):
    res_h = img_h - crop_h
    res_w = img_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w

class FSC147(Dataset):
    def __init__(self, config, split: str):
        """
        Refactored FSC147 Dataset to match T2iCount logic.
        """
        assert split in ['train', 'val', 'test', 'val_coco', 'test_coco']
        
        # --- Config Parsing ---
        self.data_dir = config['training']['data_dir']
        self.split = split
        
        # T2iCount specific params 
        self.crop_size = config['training'].get('crop_size', 384) 
        self.downsample_ratio = config['training'].get('downsample_ratio', 16) # Output stride
        self.concat_size = config['training'].get('concat_size', 224) # Size for mosaic chunks
        
        # Paths
        self.im_dir = os.path.join(self.data_dir, 'images_384_VarV2')
        self.gt_dir = os.path.join(self.data_dir, 'gt_density_map_adaptive_384_VarV2')
        self.anno_file = os.path.join(self.data_dir, 'annotation_FSC147_384.json')
        self.data_split_file = os.path.join(self.data_dir, 'Train_Test_Val_FSC_147.json')
        self.class_file = os.path.join(self.data_dir, 'ImageClasses_FSC147.txt')

        # --- Data Loading ---
        with open(self.data_split_file) as f:
            data_split = json.load(f)
            split_key = 'val' if 'val' in split else ('test' if 'test' in split else 'train')
            self.im_list = data_split[split_key]

        subset_scale = config['training'].get('subset_scale', 1.0)
        if subset_scale < 1.0:
            self.im_list = self.im_list[:int(subset_scale * len(self.im_list))]
            if split == 'train':
                random.shuffle(self.im_list)

        with open(self.anno_file) as f:
            self.annotations = json.load(f)

        # Class Mapping
        self.cls_dict = {}
        with open(self.class_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t') if '\t' in line else line.strip().split()
                key = parts[0]
                val = ' '.join(parts[1:])
                self.cls_dict[key] = val

        # 1. Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # 2. Base Transforms (T2iCount uses 0.5 mean/std, NOT ImageNet)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.im_list)

    def train_transform_density(self, img, den_map, img_attention_map):
        """
        Logic biến đổi hình học đồng bộ (Resize -> Crop -> Downsample Density -> Flip)
        """
        wd, ht = img.size
        
        # 1. Random Resize
        if random.random() >= 0.5:
            re_size = random.random() * 1 + 1 # scale [1.0, 2.0]
            wd = int(wd * re_size)
            ht = int(ht * re_size)
            img = img.resize((wd, ht), Image.Resampling.BICUBIC)

            den_map = cv2.resize(den_map, (wd, ht), interpolation=cv2.INTER_CUBIC) / (re_size ** 2)
            img_attention_map = cv2.resize(img_attention_map, (wd, ht), interpolation=cv2.INTER_NEAREST)

        # 2. Random Crop
        # Ensure image is at least crop_size
        if ht < self.crop_size or wd < self.crop_size:
             # Pad logic or simple resize up if too small (Edge case handling)
             pad_h = max(0, self.crop_size - ht)
             pad_w = max(0, self.crop_size - wd)
             img = F.pad(img, (0, 0, pad_w, pad_h))
             den_map = np.pad(den_map, ((0, pad_h), (0, pad_w)), mode='constant')
             img_attention_map = np.pad(img_attention_map, ((0, pad_h), (0, pad_w)), mode='constant')
             wd, ht = img.size

        i, j, h, w = random_crop(ht, wd, self.crop_size, self.crop_size)
        img = F.crop(img, i, j, h, w)
        den_map = den_map[i: (i + h), j: (j + w)]
        img_attention_map = img_attention_map[i: (i + h), j: (j + w)]

        # 3. Density Downsampling (SUM POOLING for T2iCount)
        den_map = den_map.reshape([h // self.downsample_ratio, self.downsample_ratio, 
                                   w // self.downsample_ratio, self.downsample_ratio]).sum(axis=(1, 3))
        
        # 4. Attention Map Resize (Nearest Neighbor)
        img_attention_map = cv2.resize(img_attention_map, (int(w / 8), int(h / 8)), interpolation=cv2.INTER_NEAREST)

        # 5. Random Flip
        if random.random() > 0.5:
            img = F.hflip(img)
            den_map = np.fliplr(den_map)
            img_attention_map = np.fliplr(img_attention_map)

        return self.transform(img), torch.from_numpy(den_map.copy()).float().unsqueeze(0), torch.from_numpy(img_attention_map.copy()).float().unsqueeze(0)

    def __getitem__(self, idx):
        im_filename = self.im_list[idx] # e.g., "100.jpg" or full path depending on json
        if not im_filename.endswith('.jpg'): im_filename = im_filename + '.jpg'
        
        # Construct paths
        im_basename = os.path.basename(im_filename)
        im_path = os.path.join(self.im_dir, im_basename)
        den_path = os.path.join(self.gt_dir, im_basename.replace('.jpg', '.npy'))

        # Load Data
        img = Image.open(im_path).convert('RGB')
        cls_name = self.cls_dict[im_basename]
        
        # Prepare Prompt Info
        prompt = cls_name
        prompt_attn_mask = torch.zeros(77) # CLIP context length
        cls_name_tokens = self.tokenizer(cls_name, add_special_tokens=False, return_tensors='pt')
        cls_name_length = cls_name_tokens['input_ids'].shape[1]
        prompt_attn_mask[1: 1 + cls_name_length] = 1 # [SOS] [tokens] ...

        if self.split == 'train':
            # Load basic density map
            try:
                den_map = np.load(den_path)
            except:
                # Fallback if npy not found (create zeros)
                wd, ht = img.size
                den_map = np.zeros((ht, wd))
            
            if random.random() > 0.5:
                wd, ht = img.size
                img_attn_map = np.ones((ht, wd))
                
                if random.random() > 0.5:
                    rand_idx = random.randint(0, len(self.im_list) - 1)
                    rand_filename = self.im_list[rand_idx]
                    rand_basename = os.path.basename(rand_filename)
                    rand_cls = self.cls_dict.get(rand_basename, "unknown")

                    out_img = img
                    if rand_cls != cls_name:
                        den_map = np.zeros((ht, wd)) 
                        prompt = rand_cls 
                        img_attn_map = np.zeros((ht, wd)) 
                        
                        prompt_attn_mask = torch.zeros(77)
                        cls_name_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
                        cls_name_length = cls_name_tokens['input_ids'].shape[1]
                        prompt_attn_mask[1: 1 + cls_name_length] = 1
                else:
                    out_img = img

            else:
                rand_indices = random.sample(range(len(self.im_list)), 3)
                rand_imgs_paths = [os.path.join(self.im_dir, os.path.basename(self.im_list[i])) for i in rand_indices]
                
                imgs_info = []
                wd, ht = img.size
                
                # 1. Process Main Image
                i, j, h, w = random_crop(ht, wd, self.concat_size, self.concat_size)
                curr_crop = F.crop(img, i, j, h, w)
                curr_den = den_map[i: (i + h), j: (j + w)]
                curr_attn = np.ones((self.concat_size, self.concat_size))
                imgs_info.append({'img': curr_crop, 'den_map': curr_den, 'img_attn': curr_attn})

                # 2. Process 3 Random Images
                for r_path in rand_imgs_paths:
                    extra_img = Image.open(r_path).convert('RGB')
                    wd, ht = extra_img.size
                    i, j, h, w = random_crop(ht, wd, self.concat_size, self.concat_size)
                    extra_crop = F.crop(extra_img, i, j, h, w)
                    
                    r_basename = os.path.basename(r_path)
                    
                    if self.cls_dict.get(r_basename) == cls_name:
                        # Cùng class -> Load density thật, Attention = 1
                        r_den_path = os.path.join(self.gt_dir, r_basename.replace('.jpg', '.npy'))
                        try:
                            extra_den = np.load(r_den_path)[i: (i + h), j: (j + w)]
                        except:
                            extra_den = np.zeros((self.concat_size, self.concat_size))
                        extra_attn = np.ones((self.concat_size, self.concat_size))
                    else:
                        # Khác class -> Density = 0, Attention = 0
                        extra_den = np.zeros((self.concat_size, self.concat_size))
                        extra_attn = np.zeros((self.concat_size, self.concat_size))
                    
                    imgs_info.append({'img': extra_crop, 'den_map': extra_den, 'img_attn': extra_attn})

                random.shuffle(imgs_info)
                
                # 3. Stitch into 2x2 Grid
                out_img = Image.new('RGB', (self.concat_size * 2, self.concat_size * 2))
                den_map = np.zeros((self.concat_size * 2, self.concat_size * 2))
                img_attn_map = np.zeros((self.concat_size * 2, self.concat_size * 2))

                positions = [
                    (0, 0), (self.concat_size, 0), 
                    (0, self.concat_size), (self.concat_size, self.concat_size)
                ]

                for idx, pos in enumerate(positions):
                    x, y = pos
                    # Paste Image
                    out_img.paste(imgs_info[idx]['img'], (x, y))
                    # Paste Density
                    den_map[y:y+self.concat_size, x:x+self.concat_size] = imgs_info[idx]['den_map']
                    # Paste Attention
                    img_attn_map[y:y+self.concat_size, x:x+self.concat_size] = imgs_info[idx]['img_attn']

            img, den_map, img_attn_map = self.train_transform_density(out_img, den_map, img_attn_map)
            
            return img, den_map, prompt, prompt_attn_mask, img_attn_map

        else:
            
            W, H = img.size
            new_H = 16 * int(H / 16)
            new_W = 16 * int(W / 16)
            img = img.resize((new_W, new_H), Image.Resampling.BICUBIC)
            
            pts = self.annotations[im_basename]['points']
            gt_count = len(pts)

            img_tensor = self.transform(img)
            
            return img_tensor, gt_count, prompt, prompt_attn_mask, im_basename.split('.')[0]

def collate_fn(batch):
    if len(batch[0]) == 5 and isinstance(batch[0][1], torch.Tensor): 
        images, densities, prompts, prompt_masks, img_attn_maps = zip(*batch)
        return {
            'image': torch.stack(images, 0),
            'density': torch.stack(densities, 0),
            'prompts': list(prompts),         # List of strings
            'prompt_masks': torch.stack(prompt_masks, 0),
            'img_attn_maps': torch.stack(img_attn_maps, 0)
        }
    else:
        images, gt_counts, prompts, prompt_masks, names = zip(*batch)
        return {
            'image': torch.stack(images, 0),
            'gt_count': torch.tensor(gt_counts),
            'prompts': list(prompts),
            'prompt_masks': torch.stack(prompt_masks, 0),
            'names': list(names)
        }
    
import logging
import os 

def logg(log_file):
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # vẫn in ra màn hình
        ]
    )