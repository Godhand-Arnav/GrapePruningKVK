import cv2
import numpy as np
import networkx as nx

class SimonitPruningEngine:
    """
    Implements 2D pruning decision heuristics based on the Simonit & Sirch
    Gentle Pruning methodology (Simonit 2014, Vid2Cuts 2024).
    """
    def __init__(self, spur_buds_retained=2, cane_buds_retained=8):
        self.spur_buds_retained = spur_buds_retained
        self.cane_buds_retained = cane_buds_retained

    def extract_branches(self, tree, root_idx=0):
        """
        Traverses the directed tree from root outward, grouping nodes into
        individual branch sequences.
        """
        branches = []
        # Find all leaves (endpoints)
        leaves = [n for n in tree.nodes() if tree.out_degree(n) == 0 and n != root_idx]
        
        for leaf in leaves:
            try:
                # Path from root to leaf
                path = nx.shortest_path(tree, source=root_idx, target=leaf)
                if len(path) >= 2:
                    branches.append(path)
            except nx.NetworkXNoPath:
                continue
                
        return branches

    def compute_pruning_suggestions(self, tree, root_idx=0):
        """
        Evaluates branches and outputs cut recommendations.
        Returns: list of dicts with:
          - branch_id
          - branch_type ('future_spur', 'fruiting_cane', 'unwanted')
          - cut_point: (x, y) 2D coordinate on image
          - cut_normal: (dx, dy) direction vector for drawing cut line
          - retained_nodes: list of node coordinates to keep
        """
        branches = self.extract_branches(tree, root_idx)
        suggestions = []
        
        if not branches:
            return suggestions
            
        # Sort branches by length (number of nodes)
        branches.sort(key=lambda b: len(b), reverse=True)
        
        # Heuristic assignment:
        # Longest branch -> Candidate Fruiting Cane (retain more buds)
        # Moderate lower branch -> Candidate Renewal Spur (retain 2 buds)
        assigned_spur = False
        
        for b_idx, path in enumerate(branches):
            nodes_info = [tree.nodes[nid] for nid in path]
            num_nodes = len(nodes_info)
            
            # Skip trivial 1-node stubs
            if num_nodes < 2:
                continue
                
            # Check if this branch should be a renewal spur
            if not assigned_spur and num_nodes >= self.spur_buds_retained + 1:
                # Renewal Spur rule: Retain 2 buds, cut between bud 2 and bud 3
                target_node_idx = self.spur_buds_retained
                p_keep = nodes_info[target_node_idx]['pos']
                
                if target_node_idx + 1 < num_nodes:
                    p_next = nodes_info[target_node_idx + 1]['pos']
                else:
                    # Extrapolate slightly past p_keep
                    p_prev = nodes_info[target_node_idx - 1]['pos']
                    p_next = (2 * p_keep[0] - p_prev[0], 2 * p_keep[1] - p_prev[1])
                    
                # Cut at midpoint between node 2 and node 3 to preserve sap flow & prevent desiccation cone
                cut_x = (p_keep[0] + p_next[0]) / 2.0
                cut_y = (p_keep[1] + p_next[1]) / 2.0
                
                # Normal vector perpendicular to the branch segment
                bx = p_next[0] - p_keep[0]
                by = p_next[1] - p_keep[1]
                blen = np.hypot(bx, by) + 1e-5
                nx_dir = -by / blen
                ny_dir = bx / blen
                
                suggestions.append({
                    'branch_idx': b_idx,
                    'role': 'Renewal Spur (Future Spur)',
                    'retained_buds': self.spur_buds_retained,
                    'cut_point': (cut_x, cut_y),
                    'cut_normal': (nx_dir, ny_dir),
                    'path_nodes': [n['pos'] for n in nodes_info[:target_node_idx + 1]]
                })
                assigned_spur = True
                
            elif b_idx == 0:
                # Primary Fruiting Cane (keep long for fruit production)
                suggestions.append({
                    'branch_idx': b_idx,
                    'role': 'Fruiting Cane',
                    'retained_buds': num_nodes,
                    'cut_point': None, # No winter cut needed
                    'path_nodes': [n['pos'] for n in nodes_info]
                })
            else:
                # Unwanted 1-year shoot / crowding branch -> basal cut
                base_pos = nodes_info[1]['pos']
                suggestions.append({
                    'branch_idx': b_idx,
                    'role': 'Removal (Thinning Cut)',
                    'retained_buds': 0,
                    'cut_point': base_pos,
                    'cut_normal': (1.0, 0.0),
                    'path_nodes': [nodes_info[0]['pos']]
                })
                
        return suggestions

    def draw_overlay(self, image, tree, suggestions, scale_factor=4.0):
        """
        Renders the color-coded decision support overlay on the input image.
        """
        vis = image.copy()
        
        # 1. Draw tree skeleton edges
        for u, v in tree.edges():
            pos_u = tree.nodes[u]['pos']
            pos_v = tree.nodes[v]['pos']
            pt1 = (int(round(pos_u[0] * scale_factor)), int(round(pos_u[1] * scale_factor)))
            pt2 = (int(round(pos_v[0] * scale_factor)), int(round(pos_v[1] * scale_factor)))
            cv2.line(vis, pt1, pt2, (200, 200, 200), 2, cv2.LINE_AA)
            
        # 2. Draw nodes
        for n, d in tree.nodes(data=True):
            pos = d['pos']
            pt = (int(round(pos[0] * scale_factor)), int(round(pos[1] * scale_factor)))
            cv2.circle(vis, pt, 5, (0, 255, 255), -1, cv2.LINE_AA)
            
        # 3. Draw pruning suggestions & cut lines
        for s in suggestions:
            role = s['role']
            cut_pt = s['cut_point']
            
            if role == 'Renewal Spur (Future Spur)':
                # Highlight retained spur path in Blue
                for i in range(len(s['path_nodes']) - 1):
                    p1 = (int(round(s['path_nodes'][i][0] * scale_factor)), 
                          int(round(s['path_nodes'][i][1] * scale_factor)))
                    p2 = (int(round(s['path_nodes'][i+1][0] * scale_factor)), 
                          int(round(s['path_nodes'][i+1][1] * scale_factor)))
                    cv2.line(vis, p1, p2, (255, 120, 0), 4, cv2.LINE_AA)
                    
            elif role == 'Fruiting Cane':
                # Highlight fruiting cane in Green
                for i in range(len(s['path_nodes']) - 1):
                    p1 = (int(round(s['path_nodes'][i][0] * scale_factor)), 
                          int(round(s['path_nodes'][i][1] * scale_factor)))
                    p2 = (int(round(s['path_nodes'][i+1][0] * scale_factor)), 
                          int(round(s['path_nodes'][i+1][1] * scale_factor)))
                    cv2.line(vis, p1, p2, (0, 220, 0), 4, cv2.LINE_AA)
                    
            # Draw Cut Line (Red with scissors indicator)
            if cut_pt is not None:
                cx = int(round(cut_pt[0] * scale_factor))
                cy = int(round(cut_pt[1] * scale_factor))
                nx_dir, ny_dir = s.get('cut_normal', (1.0, 0.0))
                
                line_len = 22
                p_cut1 = (int(round(cx - nx_dir * line_len)), int(round(cy - ny_dir * line_len)))
                p_cut2 = (int(round(cx + nx_dir * line_len)), int(round(cy + ny_dir * line_len)))
                
                # Draw thick red cut marker with black outline for visibility
                cv2.line(vis, p_cut1, p_cut2, (0, 0, 0), 6, cv2.LINE_AA)
                cv2.line(vis, p_cut1, p_cut2, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1, cv2.LINE_AA)
                
                # Add text label
                cv2.putText(vis, f"CUT: {role}", (cx + 10, cy - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                
        return vis
