import os
import sys
import argparse
import cv2
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.hourglass import StackedHourglass
from pipeline.dataset import NODE_CLASSES, BRANCH_CLASSES, TOTAL_HEATMAPS
from pipeline.graph_extractor import extract_node_peaks, build_vine_tree
from pipeline.rule_engine import SimonitPruningEngine

def main():
    parser = argparse.ArgumentParser(description='PruneGuide Inference & Visual Cut Decision Support')
    parser.add_argument('--image_path', type=str, default=None, help='Path to input grapevine image')
    parser.add_argument('--checkpoint', type=str, default=os.path.join(BASE_DIR, 'checkpoints', 'best_model.pth'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(BASE_DIR, 'outputs'))
    parser.add_argument('--img_size', type=int, default=512, help='Processing resolution')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Default to a sample image if none provided
    if args.image_path is None:
        sample_img = os.path.join(BASE_DIR, 'data', 'samples', 'test', 'images', 'synthetic_test_000.jpg')
        if os.path.exists(sample_img):
            args.image_path = sample_img
        else:
            raise FileNotFoundError('Please specify --image_path')
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('='*60)
    print(f'PruneGuide Inference Engine')
    print(f'Device: {device}')
    print(f'Input Image: {args.image_path}')
    print('='*60)
    
    # 1. Load Image
    img_orig = cv2.imread(args.image_path)
    if img_orig is None:
        raise FileNotFoundError(f'Could not load image: {args.image_path}')
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]
    
    img_resized = cv2.resize(img_rgb, (args.img_size, args.img_size))
    tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    
    # 2. Load Checkpoint or Initialize
    hg_channels = 64
    if os.path.exists(args.checkpoint):
        print(f'Loading weights from: {args.checkpoint}')
        ckpt = torch.load(args.checkpoint, map_location=device)
        hg_channels = ckpt.get('hg_channels', 64)
        model = StackedHourglass(num_heatmaps=TOTAL_HEATMAPS, in_channels=3, base_channels=64, 
                                 hg_channels=hg_channels, num_stacks=2).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print('Warning: No checkpoint found, running with freshly initialized model for demonstration.')
        model = StackedHourglass(num_heatmaps=TOTAL_HEATMAPS, in_channels=3, base_channels=64, 
                                 hg_channels=hg_channels, num_stacks=2).to(device)
        
    model.eval()
    
    # 3. Model Forward Pass
    with torch.no_grad():
        preds = model(tensor)
        # Head 2 outputs final predictions: (1, TOTAL_HEATMAPS, H/4, W/4)
        final_maps = preds[-1].squeeze(0).cpu().numpy()
        
    out_h, out_w = final_maps.shape[1], final_maps.shape[2]
    
    # 4. Extract Node Peaks
    detected_nodes = []
    node_id = 0
    for idx, ntype in enumerate(NODE_CLASSES):
        hmap = final_maps[idx]
        peaks = extract_node_peaks(hmap, threshold=0.25)
        for (x, y, conf) in peaks:
            detected_nodes.append({
                'id': node_id,
                'type': ntype,
                'pos': (x, y),
                'conf': conf
            })
            node_id += 1
            
    print(f'Detected {len(detected_nodes)} vine nodes.')
    
    # 5. Extract PAF Vector Fields (Shoot branch PAFs by default)
    paf_start_idx = len(NODE_CLASSES)
    # Shoot PAF is index 3 in branch classes -> paf_start_idx + 3*2
    paf_x = final_maps[paf_start_idx + 6]
    paf_y = final_maps[paf_start_idx + 7]
    
    # 6. Reconstruct Grapevine Tree Graph via Dijkstra
    tree = build_vine_tree(detected_nodes, paf_x, paf_y)
    print(f'Constructed plant skeleton tree with {tree.number_of_nodes()} nodes and {tree.number_of_edges()} edges.')
    
    # 7. Apply Simonit & Sirch Pruning Rules
    engine = SimonitPruningEngine(spur_buds_retained=2)
    suggestions = engine.compute_pruning_suggestions(tree)
    print('\nPruning Decision Summary:')
    for s in suggestions:
        role = s['role']
        if s['cut_point'] is not None:
            cx, cy = s['cut_point']
            # Scale coordinates back to original image size
            scale_x = w_orig / out_w
            scale_y = h_orig / out_h
            print(f'  • {role}: CUT at (x={int(cx * scale_x)}, y={int(cy * scale_y)}) | Retained buds: {s["retained_buds"]}')
        else:
            print(f'  • {role}: KEEP INTACT (Fruiting Cane) | Retained buds: {s["retained_buds"]}')
            
    # 8. Render and Save Visual Guidance Overlay
    scale_factor = w_orig / out_w
    vis_overlay = engine.draw_overlay(img_orig, tree, suggestions, scale_factor=scale_factor)
    
    out_file = os.path.join(args.output_dir, 'annotated_pruning_decision.png')
    cv2.imwrite(out_file, vis_overlay)
    print('='*60)
    print(f'Visual decision support image successfully saved at:\n{out_file}')
    print('='*60)

if __name__ == '__main__':
    main()
