#!/usr/bin/env python
"""Simple Grapevine Structural Tracing & Identification Web App.

Allows uploading an unseen grapevine image and visualizes:
  - Structural segmentation overlay (trunk, cordon, cane, shoot)
  - 1px centerline skeleton
  - Graph nodes (junctions, endpoints) and edges
  - Whole-vine hierarchy tracing
  - Structural measurements (trunk/cordon lengths, cane/shoot counts, angles)
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

_ROOT = Path(__file__).resolve().parent
src_dir = str((_ROOT / "src").resolve())
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Force reload grapevine from src if shadowed by outer folder
if "grapevine" in sys.modules and not hasattr(sys.modules["grapevine"], "measurements"):
    del sys.modules["grapevine"]

from grapevine import measurements as meas
from grapevine import postprocess as pp
from grapevine import schema
from grapevine import skeletonize as skel_mod
from grapevine import trace as trace_mod
from grapevine import visualize as viz
from grapevine.graph_build import build_graph
from scripts.infer import run_inference

app = FastAPI(title="Grapevine Structural Tracing")

# Locate weights: search in runs directory or fallback to trained weights
def get_weights_path() -> str:
    possible_paths = [
        _ROOT / "runs" / "3d2cut_run" / "weights" / "best.pt",
        Path("runs/segment/grapevine/runs/3d2cut_run/weights/best.pt"),
        _ROOT.parent / "runs" / "segment" / "grapevine" / "runs" / "3d2cut_run" / "weights" / "best.pt",
        Path("yolov8n-seg.pt"),
    ]
    # Also check recursive search in runs/
    for r in [_ROOT / "runs", Path("runs")]:
        if r.exists():
            for pt in r.glob("**/weights/best.pt"):
                if pt.exists():
                    return str(pt)
    for p in possible_paths:
        if p.exists():
            return str(p)
    return "yolov8n-seg.pt"


TEMPLATE_PATH = _ROOT / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index():
    if TEMPLATE_PATH.exists():
        return HTMLResponse(content=TEMPLATE_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Index template not found</h1>", status_code=404)


@app.post("/predict")
async def predict_grapevine(file: UploadFile = File(...)):
    weights_path = get_weights_path()
    
    # Save uploaded file to temp directory
    temp_dir = _ROOT / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"upload_{int(time.time()*1000)}_{file.filename}"
    
    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)
        
    try:
        t_post_start = time.perf_counter()
        raw_masks, image, inference_ms = run_inference(weights_path, str(temp_path))
        
        # §6.1 Mask cleanup
        clean = pp.clean_masks(raw_masks)
        combined = pp.combined_binary_mask(clean)
        class_map = pp.class_label_map(clean)
        
        # §6.2 Skeletonization
        skeleton = skel_mod.skeletonize(combined)
        
        # §6.3 Graph construction
        graph = build_graph(skeleton, class_map)
        
        # §6.4 Whole-vine tracing
        vine_traces = trace_mod.trace_all_vines(graph)
        postprocess_ms = (time.perf_counter() - t_post_start) * 1000 - inference_ms
        
        # §7 Measurements & assembly
        vines_out = []
        measurements_by_vine = {}
        roots_by_vine = {}
        
        total_trunk_px = 0.0
        total_cordon_px = 0.0
        total_cane_px = 0.0
        total_shoot_px = 0.0
        total_trunk_count = 0
        total_cordon_count = 0
        total_cane_count = 0
        total_shoot_count = 0
        
        for vine_id, tr in enumerate(vine_traces):
            sub_nodes = {n for p in tr.paths for n in p["nodes"]} | {tr.root}
            sub_g = graph.subgraph(sub_nodes)
            
            node_objs = [
                schema.Node(id=n, x=attrs["x"], y=attrs["y"], type=attrs["type"])
                for n, attrs in sub_g.nodes(data=True)
            ]
            edge_objs = [
                schema.Edge(from_id=u, to_id=v, cls=data["cls"], length_px=data["length_px"],
                            pixel_path=[(x, y) for (y, x) in data.get("pixel_path", [])])
                for u, v, data in sub_g.edges(data=True)
            ]
            path_objs = [
                schema.Path(node_ids=p["nodes"], classes=p["classes"], total_length_px=p["total_length_px"])
                for p in tr.paths
            ]
            
            m = meas.compute_measurements(sub_g, tr.root)
            measurements_by_vine[vine_id] = m
            roots_by_vine[vine_id] = tr.root
            
            total_trunk_px += m.get("trunk_length_px", 0.0)
            total_cordon_px += m.get("cordon_length_px", 0.0)
            total_cane_px += m.get("cane_length_px", 0.0)
            total_shoot_px += m.get("shoot_length_px", 0.0)
            total_trunk_count += m.get("trunk_count", 0)
            total_cordon_count += m.get("cordon_count", 0)
            total_cane_count += m.get("cane_count", 0)
            total_shoot_count += m.get("shoot_count", 0)
            
            vines_out.append(schema.Vine(
                vine_id=vine_id, root_node=tr.root, nodes=node_objs,
                edges=edge_objs, paths=path_objs, measurements=m,
            ))
            
        summary_stats = {
            "vine_count": len(vines_out),
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "junction_count": sum(1 for _, a in graph.nodes(data=True) if a.get("type") == "junction"),
            "endpoint_count": sum(1 for _, a in graph.nodes(data=True) if a.get("type") == "endpoint"),
            "total_trunk_count": total_trunk_count,
            "total_cordon_count": total_cordon_count,
            "total_cane_count": total_cane_count,
            "total_shoot_count": total_shoot_count,
            "total_trunk_px": round(total_trunk_px, 2),
            "total_cordon_px": round(total_cordon_px, 2),
            "total_cane_px": round(total_cane_px, 2),
            "total_shoot_px": round(total_shoot_px, 2),
            "total_length_px": round(total_trunk_px + total_cordon_px + total_cane_px + total_shoot_px, 2),
        }

        payload = schema.build_output(
            image_filename=file.filename,
            image_dims=(image.shape[1], image.shape[0]),
            vines=vines_out,
            model_name=Path(weights_path).stem,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
        )
        payload["summary"] = summary_stats
        
        # Render overlay PNG in-memory
        overlay_out_path = temp_dir / f"overlay_{int(time.time()*1000)}.png"
        viz.render_overlay_png(
            image=image, class_map=class_map, skeleton=skeleton, graph=graph,
            measurements_by_vine=measurements_by_vine, roots=roots_by_vine,
            out_path=str(overlay_out_path),
        )
        
        with open(overlay_out_path, "rb") as f:
            png_bytes = f.read()
        b64_png = base64.b64encode(png_bytes).decode("utf-8")
        
        # Clean up temporary files
        try:
            temp_path.unlink(missing_ok=True)
            overlay_out_path.unlink(missing_ok=True)
        except Exception:
            pass
            
        return {
            "processing": payload["processing"],
            "summary": summary_stats,
            "vines": [v.to_dict(include_pixel_path=False) for v in vines_out],
            "overlay_png_base64": b64_png,
            "result_json": payload,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})


def main():
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Grapevine web app on http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
