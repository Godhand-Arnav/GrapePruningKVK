import os
import json
import random
import cv2
import numpy as np

def draw_synthetic_grapevine(width=1024, height=1024):
    """
    Generates a realistic synthetic grapevine image and its corresponding
    node/branch graph annotations in the 3D2cut format.
    """
    # Background: textured outdoor vineyard soil/sky tone
    img = np.ones((height, width, 3), dtype=np.uint8) * 190
    # Add subtle background noise/gradient
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Trellis wire (horizontal line across middle)
    wire_y = height // 2
    cv2.line(img, (0, wire_y), (width, wire_y), (140, 140, 140), 3)
    
    nodes = []
    edges = []
    node_id_counter = 0
    
    # 1. Root Crown & Trunk
    root_x = width // 2 + random.randint(-40, 40)
    root_y = height - 80
    nodes.append({'id': node_id_counter, 'x': root_x, 'y': root_y, 'type': 'root_crown'})
    root_id = node_id_counter
    node_id_counter += 1
    
    # Trunk nodes
    curr_x, curr_y = root_x, root_y
    trunk_len = 3
    trunk_node_ids = [root_id]
    
    for _ in range(trunk_len):
        next_x = curr_x + random.randint(-15, 15)
        next_y = curr_y - random.randint(70, 100)
        nodes.append({'id': node_id_counter, 'x': next_x, 'y': next_y, 'type': 'trunk_node'})
        edges.append({'p1': (curr_x, curr_y), 'p2': (next_x, next_y), 'type': 'trunk'})
        cv2.line(img, (curr_x, curr_y), (next_x, next_y), (40, 60, 90), 22)
        curr_x, curr_y = next_x, next_y
        trunk_node_ids.append(node_id_counter)
        node_id_counter += 1
        
    head_x, head_y = curr_x, curr_y
    
    # 2. Left Branch: Renewal Spur (Courson)
    courson_x, courson_y = head_x, head_y
    for b in range(4):
        next_x = courson_x - random.randint(40, 65)
        next_y = courson_y - random.randint(50, 80)
        ntype = 'courson_node' if b == 0 else 'shoot_node'
        nodes.append({'id': node_id_counter, 'x': next_x, 'y': next_y, 'type': ntype})
        edges.append({'p1': (courson_x, courson_y), 'p2': (next_x, next_y), 'type': 'courson' if b == 0 else 'shoot'})
        cv2.line(img, (courson_x, courson_y), (next_x, next_y), (50, 90, 140), 10 if b == 0 else 6)
        # Draw small bud node
        cv2.circle(img, (next_x, next_y), 5, (30, 50, 80), -1)
        courson_x, courson_y = next_x, next_y
        node_id_counter += 1
        
    # 3. Right Branch: Fruiting Cane
    cane_x, cane_y = head_x, head_y
    for b in range(6):
        next_x = cane_x + random.randint(50, 80)
        next_y = cane_y + random.randint(-15, 25) # Follows trellis wire horizontally
        nodes.append({'id': node_id_counter, 'x': next_x, 'y': next_y, 'type': 'cane_node' if b < 2 else 'shoot_node'})
        edges.append({'p1': (cane_x, cane_y), 'p2': (next_x, next_y), 'type': 'cane' if b < 2 else 'shoot'})
        cv2.line(img, (cane_x, cane_y), (next_x, next_y), (60, 110, 160), 8 if b < 2 else 5)
        cv2.circle(img, (next_x, next_y), 5, (30, 50, 80), -1)
        cane_x, cane_y = next_x, next_y
        node_id_counter += 1
        
    annotation = {
        'width': width,
        'height': height,
        'nodes': nodes,
        'edges': edges
    }
    
    return img, annotation

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    samples_dir = os.path.join(base_dir, 'data', 'samples')
    
    for split in ['train', 'test']:
        img_dir = os.path.join(samples_dir, split, 'images')
        ann_dir = os.path.join(samples_dir, split, 'annotations')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        
        count = 12 if split == 'train' else 4
        print(f'Generating {count} synthetic samples for {split}...')
        for i in range(count):
            img, ann = draw_synthetic_grapevine()
            sample_id = f'synthetic_{split}_{i:03d}'
            
            cv2.imwrite(os.path.join(img_dir, f'{sample_id}.jpg'), img)
            with open(os.path.join(ann_dir, f'{sample_id}.json'), 'w') as f:
                json.dump(ann, f, indent=2)
                
    print(f'Synthetic dataset successfully created at: {samples_dir}')

if __name__ == '__main__':
    main()
