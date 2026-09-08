import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

# Define canonical classes
NODE_CLASSES = [
    'root_crown',
    'trunk_node',
    'courson_node',
    'cane_node',
    'shoot_node'
]

BRANCH_CLASSES = [
    'trunk',
    'courson',
    'cane',
    'shoot',
    'lateral_shoot'
]

# Total heatmaps: 5 node classes + (5 branch classes * 2 PAF components x, y) = 15 channels
TOTAL_HEATMAPS = len(NODE_CLASSES) + len(BRANCH_CLASSES) * 2

def generate_gaussian_heatmap(height, width, points, sigma):
    """
    Generates a 2D heatmap with Gaussian peaks at given points.
    Formula: M(p) = max_j exp( - ||p - x_j||^2 / sigma^2 )
    """
    heatmap = np.zeros((height, width), dtype=np.float32)
    if len(points) == 0:
        return heatmap
        
    y_coords, x_coords = np.ogrid[:height, :width]
    for (x, y) in points:
        # Distance squared
        d2 = (x_coords - x)**2 + (y_coords - y)**2
        # Gaussian blob
        blob = np.exp(-d2 / (sigma**2 + 1e-6))
        heatmap = np.maximum(heatmap, blob)
        
    return heatmap

def generate_vector_field(height, width, segments, limb_width=4.0):
    """
    Generates a 2-channel Vector Affinity Field (PAF) along line segments.
    segments: list of ((x1, y1), (x2, y2))
    Formula: v(x1, x2) = (x2 - x1) / ||x2 - x1|| for points within limb_width
    """
    paf_x = np.zeros((height, width), dtype=np.float32)
    paf_y = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    
    for (p1, p2) in segments:
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        
        dx = x2 - x1
        dy = y2 - y1
        length = np.hypot(dx, dy)
        if length < 1e-4:
            continue
            
        ux = dx / length
        uy = dy / length
        
        # Draw anti-aliased line onto temporary mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.line(mask, (int(round(x1)), int(round(y1))), 
                       (int(round(x2)), int(round(y2))), 255, int(round(limb_width * 2)))
        
        indices = mask > 0
        paf_x[indices] += ux
        paf_y[indices] += uy
        counts[indices] += 1
        
    # Average overlapping vector fields
    nonzero = counts > 0
    paf_x[nonzero] /= counts[nonzero]
    paf_y[nonzero] /= counts[nonzero]
    
    return paf_x, paf_y

class GrapevineDataset(Dataset):
    """
    Dataset for loading grapevine images and generating multi-stage heatmaps
    compatible with the 3D2cut dataset format.
    """
    def __init__(self, data_dir, split='train', target_size=(512, 512), sigma1=15.0, sigma2=6.0):
        self.data_dir = data_dir
        self.split = split
        self.target_size = target_size # (W, H)
        self.sigma1 = sigma1 # Stage 1 coarse sigma
        self.sigma2 = sigma2 # Stage 2 fine sigma
        
        self.images_dir = os.path.join(data_dir, split, 'images')
        self.ann_dir = os.path.join(data_dir, split, 'annotations')
        
        if os.path.exists(self.images_dir):
            self.sample_ids = [os.path.splitext(f)[0] for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        else:
            self.sample_ids = []
            
    def __len__(self):
        return len(self.sample_ids)
        
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        img_path = os.path.join(self.images_dir, f'{sample_id}.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.images_dir, f'{sample_id}.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.images_dir, f'{sample_id}.jpeg')
            
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'Failed to read image at {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = img.shape[:2]
        target_w, target_h = self.target_size
        img_resized = cv2.resize(img, (target_w, target_h))
        
        # Load annotation
        ann_path = os.path.join(self.ann_dir, f'{sample_id}.json')
        nodes_dict = {cls_name: [] for cls_name in NODE_CLASSES}
        branches_dict = {cls_name: [] for cls_name in BRANCH_CLASSES}
        
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                ann = json.load(f)
                
            scale_x = (target_w / 4) / orig_w
            scale_y = (target_h / 4) / orig_h
            
            # Extract nodes scaled to output feature map (W/4, H/4)
            for node in ann.get('nodes', []):
                ntype = node.get('type', 'shoot_node')
                if ntype in nodes_dict:
                    x = node['x'] * scale_x
                    y = node['y'] * scale_y
                    nodes_dict[ntype].append((x, y))
                    
            # Extract limbs scaled to output feature map
            for edge in ann.get('edges', []):
                btype = edge.get('type', 'shoot')
                if btype in branches_dict:
                    p1 = (edge['p1'][0] * scale_x, edge['p1'][1] * scale_y)
                    p2 = (edge['p2'][0] * scale_x, edge['p2'][1] * scale_y)
                    branches_dict[btype].append((p1, p2))
                    
        out_h, out_w = target_h // 4, target_w // 4
        
        # Generate stage 1 and stage 2 ground-truth stacks
        heatmaps_s1 = []
        heatmaps_s2 = []
        
        # 1. Node heatmaps
        for ntype in NODE_CLASSES:
            pts = nodes_dict[ntype]
            m1 = generate_gaussian_heatmap(out_h, out_w, pts, self.sigma1)
            m2 = generate_gaussian_heatmap(out_h, out_w, pts, self.sigma2)
            heatmaps_s1.append(m1)
            heatmaps_s2.append(m2)
            
        # 2. Branch vector affinity fields (PAFs)
        for btype in BRANCH_CLASSES:
            segs = branches_dict[btype]
            px, py = generate_vector_field(out_h, out_w, segs)
            heatmaps_s1.append(px)
            heatmaps_s1.append(py)
            heatmaps_s2.append(px)
            heatmaps_s2.append(py)
            
        # Convert to PyTorch tensors
        # Image: (3, H, W) normalized to [0, 1]
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        # Heatmap stacks: (TOTAL_HEATMAPS, H/4, W/4)
        target_s1 = torch.from_numpy(np.stack(heatmaps_s1, axis=0)).float()
        target_s2 = torch.from_numpy(np.stack(heatmaps_s2, axis=0)).float()
        
        return img_tensor, target_s1, target_s2, sample_id
