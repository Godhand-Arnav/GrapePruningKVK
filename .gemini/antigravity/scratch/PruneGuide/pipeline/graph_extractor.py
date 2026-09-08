import numpy as np
import scipy.ndimage as ndimage
import networkx as nx

def extract_node_peaks(heatmap, threshold=0.35, min_distance=5):
    """
    Extracts 2D (x, y) coordinates of node peaks from a predicted heatmap
    using local maximum suppression.
    """
    # Filter by threshold
    mask = heatmap > threshold
    if not np.any(mask):
        return []
        
    # Find local maximum filter
    footprint = np.ones((min_distance * 2 + 1, min_distance * 2 + 1))
    local_max = ndimage.maximum_filter(heatmap, footprint=footprint) == heatmap
    peaks_mask = local_max & mask
    
    # Extract labeled components to get centroids
    labeled, num_features = ndimage.label(peaks_mask)
    if num_features == 0:
        return []
        
    slices = ndimage.find_objects(labeled)
    peaks = []
    for s in slices:
        sub_map = heatmap[s]
        local_y, local_x = np.unravel_index(np.argmax(sub_map), sub_map.shape)
        y = s[0].start + local_y
        x = s[1].start + local_x
        peaks.append((float(x), float(y), float(heatmap[y, x])))
        
    return peaks

def compute_edge_resistivity(child_pt, parent_pt, paf_x, paf_y, num_samples=10):
    """
    Computes resistivity R_e = (1 - A) * ||v_cp||
    where A is the average cosine alignment between the candidate edge vector
    and the predicted vector field.
    """
    xc, yc = child_pt[:2]
    xp, yp = parent_pt[:2]
    
    dx = xp - xc
    dy = yp - yc
    length = np.hypot(dx, dy)
    if length < 1e-4:
        return 0.0
        
    # Unit vector from child to parent
    vx = dx / length
    vy = dy / length
    
    h, w = paf_x.shape
    u_vals = np.linspace(0.0, 1.0, num_samples)
    alignments = []
    
    for u in u_vals:
        px = int(np.clip(round((1 - u) * xc + u * xp), 0, w - 1))
        py = int(np.clip(round((1 - u) * yc + u * yp), 0, h - 1))
        
        fx = paf_x[py, px]
        fy = paf_y[py, px]
        
        f_norm = np.hypot(fx, fy)
        if f_norm > 1e-3:
            # Cosine similarity
            align = (fx * vx + fy * vy) / f_norm
            alignments.append(align)
        else:
            alignments.append(0.0)
            
    avg_align = np.mean(alignments)
    # Alignment A bounded in [-1, 1], map to [0, 1]
    A = max(0.0, float(avg_align))
    resistivity = (1.0 - A + 0.05) * length
    return resistivity

def build_vine_tree(detected_nodes, paf_x, paf_y, root_crown_idx=0, max_radius=80.0):
    """
    Constructs a directed tree graph of the grapevine using Dijkstra shortest path.
    detected_nodes: list of dicts [{'id': i, 'type': 'shoot_node', 'pos': (x, y), 'conf': c}]
    """
    if len(detected_nodes) == 0:
        return nx.DiGraph()
        
    G = nx.Graph()
    for i, node in enumerate(detected_nodes):
        G.add_node(i, **node)
        
    # Build candidate edges
    for i, node_i in enumerate(detected_nodes):
        pos_i = node_i['pos']
        candidates = []
        for j, node_j in enumerate(detected_nodes):
            if i == j:
                continue
            pos_j = node_j['pos']
            dist = np.hypot(pos_i[0] - pos_j[0], pos_i[1] - pos_j[1])
            if dist <= max_radius:
                candidates.append((j, dist))
                
        # If no candidates in radius, take the single closest
        if len(candidates) == 0 and len(detected_nodes) > 1:
            dists = [(j, np.hypot(pos_i[0] - detected_nodes[j]['pos'][0], 
                                  pos_i[1] - detected_nodes[j]['pos'][1])) 
                     for j in range(len(detected_nodes)) if j != i]
            candidates.append(min(dists, key=lambda x: x[1]))
            
        for (j, dist) in candidates:
            # Calculate resistivity
            r_weight = compute_edge_resistivity(pos_i, detected_nodes[j]['pos'], paf_x, paf_y)
            G.add_edge(i, j, weight=r_weight)
            
    # Compute shortest path from each node to root crown
    tree = nx.DiGraph()
    for i, node in enumerate(detected_nodes):
        tree.add_node(i, **node)
        
    root = root_crown_idx if root_crown_idx < len(detected_nodes) else 0
    
    for i in range(len(detected_nodes)):
        if i == root:
            continue
        try:
            path = nx.shortest_path(G, source=i, target=root, weight='weight')
            # path is [i, ..., parent, root]
            parent = path[1]
            tree.add_edge(parent, i, weight=G[parent][i]['weight'])
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
            
    return tree
