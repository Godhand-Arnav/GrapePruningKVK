"""PRD §8 — Output JSON schema.

Builds the exact structure specified in the PRD: version, image, image_dims,
vines[] (nodes/edges/paths/measurements), processing timings.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _native(v):
    """Coerce numpy scalar types to plain Python types so json.dumps doesn't
    choke on np.int64/np.float64 leaking in from array-derived coordinates
    and lengths."""
    if hasattr(v, "item"):
        return v.item()
    return v


SCHEMA_VERSION = "1.0"

# pixel_path is large — only kept if the final JSON stays under this many bytes.
MAX_OUTPUT_BYTES = 10 * 1024 * 1024  # 10MB, per §8


@dataclass
class Node:
    id: int
    x: int
    y: int
    type: str  # "junction" | "endpoint"

    def to_dict(self) -> dict:
        return {"id": _native(self.id), "x": _native(self.x), "y": _native(self.y), "type": self.type}


@dataclass
class Edge:
    from_id: int
    to_id: int
    cls: str  # trunk | cordon | cane | shoot
    length_px: float
    pixel_path: list[tuple[int, int]] = field(default_factory=list)

    def to_dict(self, include_pixel_path: bool) -> dict:
        d = {
            "from": _native(self.from_id),
            "to": _native(self.to_id),
            "class": self.cls,
            "length_px": round(_native(self.length_px), 2),
        }
        if include_pixel_path:
            d["pixel_path"] = [[_native(a), _native(b)] for a, b in self.pixel_path]
        return d


@dataclass
class Path:
    node_ids: list[int]
    classes: list[str]
    total_length_px: float

    def to_dict(self) -> dict:
        return {
            "nodes": [_native(n) for n in self.node_ids],
            "classes": self.classes,
            "total_length_px": round(_native(self.total_length_px), 2),
        }


@dataclass
class Vine:
    vine_id: int
    root_node: int
    nodes: list[Node]
    edges: list[Edge]
    paths: list[Path]
    measurements: dict[str, Any]

    def to_dict(self, include_pixel_path: bool) -> dict:
        return {
            "vine_id": _native(self.vine_id),
            "root_node": _native(self.root_node),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict(include_pixel_path) for e in self.edges],
            "paths": [p.to_dict() for p in self.paths],
            "measurements": self.measurements,
        }


def build_output(
    image_filename: str,
    image_dims: tuple[int, int],
    vines: list[Vine],
    model_name: str,
    inference_ms: float,
    postprocess_ms: float,
) -> dict:
    """Assemble the full PRD §8 JSON. Drops pixel_path if the payload would
    exceed MAX_OUTPUT_BYTES, per the spec's "omit if larger" rule."""

    total_ms = inference_ms + postprocess_ms

    def _assemble(include_pixel_path: bool) -> dict:
        return {
            "version": SCHEMA_VERSION,
            "image": image_filename,
            "image_dims": list(image_dims),
            "vines": [v.to_dict(include_pixel_path) for v in vines],
            "processing": {
                "model": model_name,
                "inference_ms": round(inference_ms, 2),
                "postprocess_ms": round(postprocess_ms, 2),
                "total_ms": round(total_ms, 2),
            },
        }

    payload = _assemble(include_pixel_path=True)
    size = len(json.dumps(payload, default=_native).encode("utf-8"))
    if size > MAX_OUTPUT_BYTES:
        payload = _assemble(include_pixel_path=False)
    return payload


def write_output(payload: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_native)
