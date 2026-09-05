#!/usr/bin/env python
"""End-to-end pipeline: single image -> PRD §8 JSON + §9 overlay PNG.

    infer (YOLO-seg) -> postprocess (§6.1) -> skeletonize (§6.2)
        -> graph_build (§6.3) -> trace (§6.4) -> measurements (§7)
        -> schema (§8) + visualize (§9)

Usage:
    python scripts/pipeline.py --weights runs/train1/weights/best.pt \
        --image data/raw/vine_0231.jpg --out-dir outputs/vine_0231
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from grapevine import measurements as meas
from grapevine import postprocess as pp
from grapevine import schema
from grapevine import skeletonize as skel_mod
from grapevine import trace as trace_mod
from grapevine import visualize as viz
from grapevine.graph_build import build_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Import here so `python scripts/pipeline.py --help` doesn't require ultralytics installed
    from scripts.infer import run_inference

    t_post_start = time.perf_counter()

    raw_masks, image, inference_ms = run_inference(args.weights, args.image)

    # §6.1 mask cleanup
    clean = pp.clean_masks(raw_masks)
    combined = pp.combined_binary_mask(clean)
    class_map = pp.class_label_map(clean)

    # §6.2 skeletonization
    skeleton = skel_mod.skeletonize(combined)

    # §6.3 graph construction
    graph = build_graph(skeleton, class_map)

    # §6.4 whole-vine tracing (handles multiple vines)
    vine_traces = trace_mod.trace_all_vines(graph)

    postprocess_ms = (time.perf_counter() - t_post_start) * 1000 - inference_ms

    # §7 measurements + §8 schema assembly, per traced vine
    vines_out = []
    measurements_by_vine = {}
    roots_by_vine = {}
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

        vines_out.append(schema.Vine(
            vine_id=vine_id, root_node=tr.root, nodes=node_objs,
            edges=edge_objs, paths=path_objs, measurements=m,
        ))

    payload = schema.build_output(
        image_filename=Path(args.image).name,
        image_dims=(image.shape[1], image.shape[0]),
        vines=vines_out,
        model_name=Path(args.weights).stem,
        inference_ms=inference_ms,
        postprocess_ms=postprocess_ms,
    )
    json_path = args.out_dir / "result.json"
    schema.write_output(payload, str(json_path))

    # §9 visualization
    png_path = args.out_dir / "overlay.png"
    viz.render_overlay_png(
        image=image, class_map=class_map, skeleton=skeleton, graph=graph,
        measurements_by_vine=measurements_by_vine, roots=roots_by_vine,
        out_path=str(png_path),
    )

    n_vines = len(vines_out)
    total_ms = inference_ms + postprocess_ms
    print(f"Done: {n_vines} vine(s) traced. inference={inference_ms:.1f}ms "
          f"postprocess={postprocess_ms:.1f}ms total={total_ms:.1f}ms")
    print(f"  -> {json_path}")
    print(f"  -> {png_path}")


if __name__ == "__main__":
    main()
