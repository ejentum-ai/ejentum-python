"""
Ejentum Multi-Modal Visual & Layout Invariants.
Provides deterministic verification for spatial layouts, bounding-box consistency,
diagram semantic referential integrity, and frame sequence drift for vision agents.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union


@dataclass
class BoundingBox:
    """Represents a 2D bounding box with normalized [ymin, xmin, ymax, xmax] coordinates in [0.0, 1.0]."""
    ymin: float
    xmin: float
    ymax: float
    xmax: float
    label: Optional[str] = None
    confidence: Optional[float] = None
    element_id: Optional[str] = None

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.ymin + self.ymax) / 2.0, (self.xmin + self.xmax) / 2.0)

    @property
    def aspect_ratio(self) -> float:
        h = self.height
        return (self.width / h) if h > 1e-6 else 0.0

    def is_valid(self, allow_pixel_coords: bool = False) -> Tuple[bool, Optional[str]]:
        """Validate coordinate ordering and boundary bounds."""
        if math.isnan(self.ymin) or math.isnan(self.xmin) or math.isnan(self.ymax) or math.isnan(self.xmax):
            return False, "Coordinates contain NaN values."
        if self.ymin >= self.ymax:
            return False, f"Inverted vertical coordinates: ymin ({self.ymin}) >= ymax ({self.ymax})."
        if self.xmin >= self.xmax:
            return False, f"Inverted horizontal coordinates: xmin ({self.xmin}) >= xmax ({self.xmax})."

        if not allow_pixel_coords:
            if not (0.0 <= self.ymin <= 1.0 and 0.0 <= self.xmin <= 1.0 and 0.0 <= self.ymax <= 1.0 and 0.0 <= self.xmax <= 1.0):
                return False, f"Normalized coordinates out of [0.0, 1.0] range: y=[{self.ymin}, {self.ymax}], x=[{self.xmin}, {self.xmax}]."

        if self.area <= 0.0:
            return False, "Degenerate zero-area bounding box."

        return True, None

    def intersection_area(self, other: "BoundingBox") -> float:
        """Calculate intersection area with another bounding box."""
        inter_ymin = max(self.ymin, other.ymin)
        inter_xmin = max(self.xmin, other.xmin)
        inter_ymax = min(self.ymax, other.ymax)
        inter_xmax = min(self.xmax, other.xmax)

        if inter_ymin < inter_ymax and inter_xmin < inter_xmax:
            return (inter_ymax - inter_ymin) * (inter_xmax - inter_xmin)
        return 0.0

    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection-over-Union (IoU) with another bounding box."""
        inter = self.intersection_area(other)
        if inter <= 0.0:
            return 0.0
        union = self.area + other.area - inter
        return (inter / union) if union > 0.0 else 0.0

    def overlap_ratio(self, other: "BoundingBox") -> float:
        """Calculate overlap ratio relative to the smaller bounding box area."""
        inter = self.intersection_area(other)
        min_area = min(self.area, other.area)
        return (inter / min_area) if min_area > 0.0 else 0.0

    def is_above(self, other: "BoundingBox", tolerance: float = 0.01) -> bool:
        """True if self is vertically positioned above other."""
        return self.ymax <= (other.ymin + tolerance)

    def is_below(self, other: "BoundingBox", tolerance: float = 0.01) -> bool:
        """True if self is vertically positioned below other."""
        return self.ymin >= (other.ymax - tolerance)

    def is_left_of(self, other: "BoundingBox", tolerance: float = 0.01) -> bool:
        """True if self is horizontally positioned to the left of other."""
        return self.xmax <= (other.xmin + tolerance)

    def is_right_of(self, other: "BoundingBox", tolerance: float = 0.01) -> bool:
        """True if self is horizontally positioned to the right of other."""
        return self.xmin >= (other.xmax - tolerance)

    def contains(self, other: "BoundingBox", tolerance: float = 0.01) -> bool:
        """True if self completely encloses other."""
        return (
            self.ymin <= (other.ymin + tolerance)
            and self.xmin <= (other.xmin + tolerance)
            and self.ymax >= (other.ymax - tolerance)
            and self.xmax >= (other.xmax - tolerance)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ymin": self.ymin,
            "xmin": self.xmin,
            "ymax": self.ymax,
            "xmax": self.xmax,
            "label": self.label,
            "confidence": self.confidence,
            "element_id": self.element_id,
            "width": self.width,
            "height": self.height,
            "area": self.area,
        }

    @classmethod
    def from_dict(cls, data: Any, default_label: Optional[str] = None) -> "BoundingBox":
        if isinstance(data, (list, tuple)):
            if len(data) >= 4:
                return cls(
                    ymin=float(data[0]),
                    xmin=float(data[1]),
                    ymax=float(data[2]),
                    xmax=float(data[3]),
                    label=default_label,
                )
            raise ValueError(f"Expected at least 4 coordinates, got {len(data)}")

        if isinstance(data, dict):
            label = data.get("label") or data.get("name") or data.get("class") or default_label
            elem_id = data.get("element_id") or data.get("id")
            conf = data.get("confidence") or data.get("score")

            if "ymin" in data and "xmin" in data and "ymax" in data and "xmax" in data:
                return cls(
                    ymin=float(data["ymin"]),
                    xmin=float(data["xmin"]),
                    ymax=float(data["ymax"]),
                    xmax=float(data["xmax"]),
                    label=label,
                    confidence=float(conf) if conf is not None else None,
                    element_id=str(elem_id) if elem_id else None,
                )
            elif "box_2d" in data and isinstance(data["box_2d"], (list, tuple)):
                coords = [float(v) for v in data["box_2d"][:4]]
                # Normalizing if provided in 0-1000 scale
                if any(c > 1.0 for c in coords):
                    coords = [c / 1000.0 for c in coords]
                return cls(
                    ymin=coords[0],
                    xmin=coords[1],
                    ymax=coords[2],
                    xmax=coords[3],
                    label=label,
                    confidence=float(conf) if conf is not None else None,
                    element_id=str(elem_id) if elem_id else None,
                )
            elif "x" in data and "y" in data and ("width" in data or "w" in data) and ("height" in data or "h" in data):
                x = float(data["x"])
                y = float(data["y"])
                w = float(data.get("width", data.get("w", 0.0)))
                h = float(data.get("height", data.get("h", 0.0)))
                return cls(
                    ymin=y,
                    xmin=x,
                    ymax=y + h,
                    xmax=x + w,
                    label=label,
                    confidence=float(conf) if conf is not None else None,
                    element_id=str(elem_id) if elem_id else None,
                )

        raise ValueError(f"Unrecognized bounding box format: {data}")


@dataclass
class SpatialVerificationResult:
    """Result of verifying 2D spatial layouts and bounding boxes."""
    is_valid: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    element_count: int = 0
    collision_pairs: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "violations": self.violations,
            "warnings": self.warnings,
            "element_count": self.element_count,
            "collision_pairs": self.collision_pairs,
            "metadata": self.metadata,
        }


class SpatialLayoutVerifier:
    """Verifies layout integrity, collision bounds, and spatial placement constraints."""

    def __init__(
        self,
        max_overlap_iou: float = 0.20,
        max_overlap_ratio: float = 0.35,
        allow_containment: bool = True,
        allow_pixel_coords: bool = False,
    ):
        self.max_overlap_iou = max_overlap_iou
        self.max_overlap_ratio = max_overlap_ratio
        self.allow_containment = allow_containment
        self.allow_pixel_coords = allow_pixel_coords

    def verify(
        self,
        elements: List[Any],
        directional_constraints: Optional[List[Dict[str, Any]]] = None,
    ) -> SpatialVerificationResult:
        """Validate layout elements and constraints."""
        violations: List[str] = []
        warnings: List[str] = []
        collision_pairs: List[Dict[str, Any]] = []

        boxes: List[BoundingBox] = []
        element_map: Dict[str, BoundingBox] = {}

        # 1. Parse and validate individual bounding boxes
        for idx, item in enumerate(elements):
            try:
                box = BoundingBox.from_dict(item, default_label=f"element_{idx}")
            except Exception as exc:
                violations.append(f"InvalidBoxFormat: Element index {idx} failed parsing: {exc}")
                continue

            valid, err = box.is_valid(allow_pixel_coords=self.allow_pixel_coords)
            if not valid:
                violations.append(f"InvalidBoxCoordinates: Element '{box.label or idx}' has invalid coordinates: {err}")
                continue

            boxes.append(box)
            key = box.element_id or box.label or f"element_{idx}"
            element_map[key] = box
            if box.label:
                element_map[box.label] = box

        # 2. Check collision and illegal overlap invariants
        n = len(boxes)
        for i in range(n):
            for j in range(i + 1, n):
                b1, b2 = boxes[i], boxes[j]
                iou_val = b1.iou(b2)
                overlap_ratio = b1.overlap_ratio(b2)

                if iou_val > 0.0 or overlap_ratio > 0.0:
                    # Check if one container legitimately contains the other
                    is_contained = b1.contains(b2) or b2.contains(b1)
                    if self.allow_containment and is_contained:
                        continue

                    if iou_val > self.max_overlap_iou or overlap_ratio > self.max_overlap_ratio:
                        name1 = b1.element_id or b1.label or f"elem_{i}"
                        name2 = b2.element_id or b2.label or f"elem_{j}"
                        col = {
                            "element1": name1,
                            "element2": name2,
                            "iou": round(iou_val, 4),
                            "overlap_ratio": round(overlap_ratio, 4),
                        }
                        collision_pairs.append(col)
                        violations.append(
                            f"SpatialCollisionViolation: Elements '{name1}' and '{name2}' overlap illegally "
                            f"(IoU={iou_val:.3f}, ratio={overlap_ratio:.3f})."
                        )

        # 3. Check directional constraints
        if directional_constraints:
            for constraint in directional_constraints:
                elem_key = constraint.get("element")
                target_box = element_map.get(elem_key)
                if not target_box:
                    warnings.append(f"DirectionalConstraintWarning: Element '{elem_key}' not found in layout.")
                    continue

                if "must_be_above" in constraint:
                    ref_key = constraint["must_be_above"]
                    ref_box = element_map.get(ref_key)
                    if ref_box and not target_box.is_above(ref_box):
                        violations.append(
                            f"DirectionalConstraintViolation: Element '{elem_key}' (ymax={target_box.ymax:.3f}) "
                            f"must be above '{ref_key}' (ymin={ref_box.ymin:.3f})."
                        )

                if "must_be_below" in constraint:
                    ref_key = constraint["must_be_below"]
                    ref_box = element_map.get(ref_key)
                    if ref_box and not target_box.is_below(ref_box):
                        violations.append(
                            f"DirectionalConstraintViolation: Element '{elem_key}' (ymin={target_box.ymin:.3f}) "
                            f"must be below '{ref_key}' (ymax={ref_box.ymax:.3f})."
                        )

                if "must_be_left_of" in constraint:
                    ref_key = constraint["must_be_left_of"]
                    ref_box = element_map.get(ref_key)
                    if ref_box and not target_box.is_left_of(ref_box):
                        violations.append(
                            f"DirectionalConstraintViolation: Element '{elem_key}' (xmax={target_box.xmax:.3f}) "
                            f"must be to the left of '{ref_key}' (xmin={ref_box.xmin:.3f})."
                        )

                if "must_be_right_of" in constraint:
                    ref_key = constraint["must_be_right_of"]
                    ref_box = element_map.get(ref_key)
                    if ref_box and not target_box.is_right_of(ref_box):
                        violations.append(
                            f"DirectionalConstraintViolation: Element '{elem_key}' (xmin={target_box.xmin:.3f}) "
                            f"must be to the right of '{ref_key}' (xmax={ref_box.xmax:.3f})."
                        )

        return SpatialVerificationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            element_count=len(boxes),
            collision_pairs=collision_pairs,
            metadata={"parsed_elements": len(boxes)},
        )


@dataclass
class DiagramVerificationResult:
    """Result of verifying diagram and flowchart semantic referential integrity."""
    is_valid: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    nodes: List[str] = field(default_factory=list)
    edges: List[Dict[str, str]] = field(default_factory=list)
    dangling_edges: List[Dict[str, str]] = field(default_factory=list)
    cycles_detected: List[List[str]] = field(default_factory=list)
    isolated_nodes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "violations": self.violations,
            "warnings": self.warnings,
            "nodes": self.nodes,
            "edges": self.edges,
            "dangling_edges": self.dangling_edges,
            "cycles_detected": self.cycles_detected,
            "isolated_nodes": self.isolated_nodes,
        }


class DiagramArtifactTrap:
    """Traps hallucinated diagram connections, phantom nodes, dangling edges, and illegal cycles."""

    # Regex patterns to parse Mermaid, Graphviz DOT, and ASCII arrow relationships
    EDGE_PATTERNS = [
        re.compile(r"([A-Za-z0-9_\-\.]+)\s*(?:-->|==>|-.->|->|=>)\s*([A-Za-z0-9_\-\.]+)"),
        re.compile(r"\[([A-Za-z0-9_\-\.\s]+)\]\s*(?:-->|->)\s*\[([A-Za-z0-9_\-\.\s]+)\]"),
        re.compile(r"([A-Za-z0-9_\-\.]+)\s*-->\|.*?\|\s*([A-Za-z0-9_\-\.]+)"),
    ]

    NODE_ID_PATTERN = re.compile(r"([A-Za-z0-9_\-\.]+)\s*(?:\[|\(|\{|\>)")

    def __init__(self, enforce_dag: bool = False, allow_isolated_nodes: bool = False):
        self.enforce_dag = enforce_dag
        self.allow_isolated_nodes = allow_isolated_nodes

    def verify(
        self,
        diagram_spec: Union[str, Dict[str, Any]],
        expected_nodes: Optional[List[str]] = None,
    ) -> DiagramVerificationResult:
        """Parse diagram text or structured graph and evaluate invariants."""
        violations: List[str] = []
        warnings: List[str] = []
        extracted_nodes: Set[str] = set()
        extracted_edges: List[Dict[str, str]] = []

        if isinstance(diagram_spec, dict):
            # Structured graph dictionary
            raw_nodes = diagram_spec.get("nodes", [])
            for n in raw_nodes:
                nid = n if isinstance(n, str) else n.get("id") or n.get("name")
                if nid:
                    extracted_nodes.add(str(nid).strip())

            raw_edges = diagram_spec.get("edges", [])
            for e in raw_edges:
                if isinstance(e, dict):
                    src = e.get("source") or e.get("from")
                    tgt = e.get("target") or e.get("to")
                    if src and tgt:
                        extracted_edges.append({"source": str(src).strip(), "target": str(tgt).strip()})
        else:
            text = str(diagram_spec)
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("%%") or line.startswith("//"):
                    continue

                for pat in self.EDGE_PATTERNS:
                    match = pat.search(line)
                    if match:
                        src = match.group(1).strip()
                        tgt = match.group(2).strip()
                        extracted_edges.append({"source": src, "target": tgt})
                        extracted_nodes.add(src)
                        extracted_nodes.add(tgt)
                        break

                for m in self.NODE_ID_PATTERN.finditer(line):
                    extracted_nodes.add(m.group(1).strip())

        # If an explicit expected_nodes manifest was provided, enforce node inventory
        dangling_edges: List[Dict[str, str]] = []
        if expected_nodes:
            expected_set = set(expected_nodes)
            for edge in extracted_edges:
                src, tgt = edge["source"], edge["target"]
                if src not in expected_set or tgt not in expected_set:
                    dangling_edges.append(edge)
                    violations.append(
                        f"DanglingEdgeViolation: Edge '{src} -> {tgt}' connects to a phantom node "
                        f"not declared in the expected node manifest."
                    )

        # Check for isolated disconnected nodes
        connected_nodes = set()
        for e in extracted_edges:
            connected_nodes.add(e["source"])
            connected_nodes.add(e["target"])

        isolated = [n for n in extracted_nodes if n not in connected_nodes]
        if isolated and not self.allow_isolated_nodes and len(extracted_edges) > 0:
            warnings.append(f"IsolatedNodesWarning: Found {len(isolated)} disconnected nodes: {isolated}")

        # Cycle detection if DAG enforcement is requested
        cycles: List[List[str]] = []
        if self.enforce_dag and extracted_edges:
            cycles = self._detect_cycles(extracted_nodes, extracted_edges)
            if cycles:
                for c in cycles:
                    violations.append(f"CycleViolation: Detected cyclic loop in DAG architecture: {' -> '.join(c)}")

        return DiagramVerificationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            nodes=sorted(list(extracted_nodes)),
            edges=extracted_edges,
            dangling_edges=dangling_edges,
            cycles_detected=cycles,
            isolated_nodes=isolated,
        )

    def _detect_cycles(self, nodes: Set[str], edges: List[Dict[str, str]]) -> List[List[str]]:
        adj: Dict[str, List[str]] = {n: [] for n in nodes}
        for e in edges:
            adj.setdefault(e["source"], []).append(e["target"])

        visited: Dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=visited
        cycles: List[List[str]] = []

        def dfs(node: str, path: List[str]):
            visited[node] = 1
            path.append(node)
            for neighbor in adj.get(node, []):
                if visited.get(neighbor) == 1:
                    idx = path.index(neighbor)
                    cycles.append(path[idx:] + [neighbor])
                elif visited.get(neighbor, 0) == 0:
                    dfs(neighbor, path)
            path.pop()
            visited[node] = 2

        for n in list(nodes):
            if visited.get(n, 0) == 0:
                dfs(n, [])

        return cycles


@dataclass
class FrameSequenceResult:
    """Result of evaluating multi-frame temporal continuity and teleportation drift."""
    is_valid: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    teleportation_events: List[Dict[str, Any]] = field(default_factory=list)
    flicker_events: List[Dict[str, Any]] = field(default_factory=list)
    max_observed_drift: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "violations": self.violations,
            "warnings": self.warnings,
            "teleportation_events": self.teleportation_events,
            "flicker_events": self.flicker_events,
            "max_observed_drift": self.max_observed_drift,
        }


class FrameSequenceDriftGuard:
    """Guards multi-frame agent trajectories against teleportation drift, flickering, and sudden scale jumps."""

    def __init__(
        self,
        max_displacement_per_frame: float = 0.40,
        max_scale_factor_jump: float = 3.0,
    ):
        self.max_displacement_per_frame = max_displacement_per_frame
        self.max_scale_factor_jump = max_scale_factor_jump

    def verify(self, frames: List[Dict[str, Any]]) -> FrameSequenceResult:
        """Evaluate temporal continuity across a sequence of frames."""
        violations: List[str] = []
        warnings: List[str] = []
        teleportation_events: List[Dict[str, Any]] = []
        flicker_events: List[Dict[str, Any]] = []
        max_drift = 0.0

        if len(frames) < 2:
            return FrameSequenceResult(is_valid=True, max_observed_drift=0.0)

        # Parse per-frame objects map: frame_idx -> {element_id -> BoundingBox}
        parsed_frames: List[Dict[str, BoundingBox]] = []
        for idx, f in enumerate(frames):
            frame_boxes = {}
            raw_objs = f.get("objects") or f.get("elements") or f.get("bounding_boxes") or []
            for item in raw_objs:
                try:
                    box = BoundingBox.from_dict(item)
                    key = box.element_id or box.label
                    if key:
                        frame_boxes[key] = box
                except Exception:
                    continue
            parsed_frames.append(frame_boxes)

        # Check frame-to-frame displacement and scale jumps
        num_frames = len(parsed_frames)
        for i in range(num_frames - 1):
            f_curr = parsed_frames[i]
            f_next = parsed_frames[i + 1]

            for key, box_curr in f_curr.items():
                if key in f_next:
                    box_next = f_next[key]
                    c1_y, c1_x = box_curr.center
                    c2_y, c2_x = box_next.center

                    # Euclidean displacement in normalized space
                    displacement = math.sqrt((c2_y - c1_y) ** 2 + (c2_x - c1_x) ** 2)
                    max_drift = max(max_drift, displacement)

                    if displacement > self.max_displacement_per_frame:
                        event = {
                            "element": key,
                            "frame_from": i,
                            "frame_to": i + 1,
                            "displacement": round(displacement, 4),
                            "threshold": self.max_displacement_per_frame,
                        }
                        teleportation_events.append(event)
                        violations.append(
                            f"TeleportationDriftViolation: Element '{key}' jumped {displacement:.3f} "
                            f"between frames {i} and {i + 1} (threshold={self.max_displacement_per_frame})."
                        )

                    # Scale jump check
                    if box_curr.area > 1e-5 and box_next.area > 1e-5:
                        scale_ratio = max(box_curr.area / box_next.area, box_next.area / box_curr.area)
                        if scale_ratio > self.max_scale_factor_jump:
                            warnings.append(
                                f"ScaleJumpWarning: Element '{key}' experienced sudden {scale_ratio:.1f}x "
                                f"area change between frames {i} and {i + 1}."
                            )

        # Flicker check: object present in frame i, missing in i+1, present in i+2
        for i in range(num_frames - 2):
            f0 = parsed_frames[i]
            f1 = parsed_frames[i + 1]
            f2 = parsed_frames[i + 2]

            for key in f0:
                if key not in f1 and key in f2:
                    event = {"element": key, "missing_frame": i + 1}
                    flicker_events.append(event)
                    warnings.append(f"FlickerWarning: Element '{key}' flickered out of frame {i + 1}.")

        return FrameSequenceResult(
            is_valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            teleportation_events=teleportation_events,
            flicker_events=flicker_events,
            max_observed_drift=round(max_drift, 4),
        )
