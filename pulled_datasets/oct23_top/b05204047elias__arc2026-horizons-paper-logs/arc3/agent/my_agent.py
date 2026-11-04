"""Just-explore graph explorer, ported to the ARC Prize 2026 ARC-AGI-3 kit.

This file is self-contained because the starter's build_notebook.py splices a
single agent file into the Kaggle notebook.

Sections
--------
1. GraphExplorer / NodeInfo and 2. FrameProcessor are copied from
   dolphin-in-a-coma/arc-agi-3-just-explore @ c2d98318ecca38c8ee9fb49b5af159e51770bad7
   (graph_explorer.py lines 15-451; agents/heuristic_agent.py lines 491-898
   without the matplotlib-only visualize_components). Changes, all marked [port]:
   every print() goes to a debug logger, and choose_edge() formats the node table
   only when reasoning is requested. Exploration logic is unchanged.
   Those sections are under the MIT License below.
3. MyAgent is new code (MIT-0, see the repository LICENSE). It replaces the 2025
   HeuristicAgent shell: arcengine types, levels_completed instead of score,
   no sleep, no LLM dependencies, per-level statistics for the JSONL logs.

MIT License (sections 1 and 2)

Copyright (c) 2025 Evgenii Rudakov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import hashlib
import logging
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple

import numpy as np
from arcengine import FrameData, GameAction, GameState

# Resolves when the ARC-AGI-3-Agents framework is on sys.path (locally and on Kaggle).
from agents.agent import Agent

logger = logging.getLogger(__name__)


def _log(*args: Any) -> None:  # [port] replaces print()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(" ".join(str(a) for a in args))


# ---------------------------------------------------------------------------
# 1. GraphExplorer (MIT, Evgenii Rudakov)
# ---------------------------------------------------------------------------
INFINITY = np.iinfo(np.int32).max


# NOTE: all data formats here chosen crudely, to be optimized later
edge_dtype = np.dtype([
    ("group", "i4"), # 0-indexed group id
    ("result", "i4"), # 1 if success, -1 if failed, 0 if not tested yet
    ("target", "U32"), # target node hash-name, "" if not tested or failed
    ("distance", "i4"), # distance to the frontier node, 0 means next node is the frontier
    ("errors", "i4"), # number of errors so far
])

def format_struct_table(arr):
    names = ("idx",) + arr.dtype.names
    cols = []
    for name in names:
        if name == "idx":
            cols.append([str(i) for i in range(len(arr))])
        else:
            cols.append([str(r[name]) for r in arr])
    widths = [max(len(n), *(len(v) for v in col)) for n, col in zip(names, cols)]
    header = " | ".join(n.ljust(w) for n, w in zip(names, widths))
    sep = "-+-".join("-"*w for w in widths)
    lines = []
    for i in range(len(arr)):
        line = " | ".join(cols[j][i].ljust(widths[j]) for j in range(len(names)))
        lines.append(line)
    return "\n".join([header, sep, *lines])

@dataclass
class NodeInfo:
    name: Hashable

    total_candidates: int # how many exist
    num_groups: int = 1 # FIXME: is never used
    active_group: int = 0

    group2remaining_candidate_ids: List[Set[int]] = field(default_factory=list)

    edge_data: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=edge_dtype))

    error_threshold: int = 3
    closed: bool = False # flips when last probe done
    distance: float | None = 0 # TODO: how is it initialized?

    def __post_init__(self):

        assert self.name is not None, "Node name must be provided"

        if self.num_groups > 1 and self.group2remaining_candidate_ids is None:
            raise ValueError("group2remaining_candidate_ids must be provided if num_groups > 1")

        if self.num_groups == 1 and self.group2remaining_candidate_ids is None:
            self.group2remaining_candidate_ids = [set(range(self.total_candidates))]

        self.group2remaining_candidate_ids = [set(r_c_ids) for r_c_ids in self.group2remaining_candidate_ids] # ensure it's a list of sets

        self.edge_data = np.zeros(self.total_candidates, dtype=edge_dtype)
            
        for group_id, remaining_candidate_ids in enumerate(self.group2remaining_candidate_ids):
            self.edge_data["group"][list(remaining_candidate_ids)] = group_id

    @property
    def has_open(self) -> bool:
        """Still hiding ≥1 untested edge?"""
        return len(self.tested) < self.total_candidates

    def record_test(self, edge_idx: int, success: int, target_node: Hashable | None = None) -> bool:

        edge_group_id = self.edge_data[edge_idx]["group"]

        assert self.edge_data["result"][edge_idx] == 0 and \
            self.edge_data["target"][edge_idx] == "" and \
            self.edge_data["distance"][edge_idx] == 0, \
            "Edge result must be untested before recording a test"

        if success == -1:
            self.edge_data["errors"][edge_idx] += 1
            if self.edge_data["errors"][edge_idx] >= self.error_threshold:
                self.edge_data["errors"][edge_idx] = 0
                new_group_id = edge_group_id + 1
                if new_group_id > self.num_groups - 1:
                    # count it as failed and move on
                    self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)
                    self.edge_data["result"][edge_idx] = -1
                    self.edge_data["distance"][edge_idx] = INFINITY
                    return True
                else:
                    self.edge_data["group"][edge_idx] = new_group_id
                    self.group2remaining_candidate_ids[new_group_id].add(edge_idx)
                    self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)
            return False

        self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)

        if success == 1:
            self.edge_data["target"][edge_idx] = str(target_node)
            self.edge_data["distance"][edge_idx] = -1 # NOTE: distance is maintained by the GraphExplorer class
            self.edge_data["result"][edge_idx] = 1
        elif success == 0:
            self.edge_data["distance"][edge_idx] = INFINITY
            self.edge_data["result"][edge_idx] = -1

        return True

    def has_open_group(self, group_id: int) -> bool:
        """Return True if this node has at least one untested edge belonging to *group_id* or below."""
        for i in range(group_id+1):
            if len(self.group2remaining_candidate_ids[i]) > 0:
                return True
        return False
    
    def __repr__(self) -> str:
        edge_data_repr = format_struct_table(self.edge_data)

        return f"""NodeInfo:
name={self.name},
total_candidates={self.total_candidates},
num_groups={self.num_groups},
distance={self.distance},
closed={self.closed},
{edge_data_repr}
"""


class GraphExplorer:

    def __init__(
        self,
        start_node: Hashable | None = None, 
        num_candidates: int | None = None, 
        group2remaining_candidate_ids: List[Set[int]] | None = None,
        n_groups: int = 1,
        verbose_level: int = 0,
        ) -> None:

        self._verbose_level = verbose_level
        self._n_groups = max(1, n_groups)

        self.reset()

    def reset(self) -> None:
        self._nodes: Dict[Hashable, NodeInfo] = {}
        self._G: Dict[Hashable, Set[Tuple[int, Hashable]]] = defaultdict(set) # (edge_idx, target_node)
        self._G_rev: Dict[Hashable, Set[Tuple[int, Hashable]]] = defaultdict(set) # (edge_idx, source_node)
        self._frontier: Set[Hashable] = set()
        self._dist: Dict[Hashable, int] = {}
        self._next: Dict[Hashable, Tuple[int, Hashable]] = {} # (edge_idx, target_node)
        self._active_group: int = 0  # current priority group

        self.suspicious_transitions: Dict[Tuple[Hashable, int, Hashable], int] = {} # (source_node, edge_idx, target_node) -> count
        self.suspicious_transitions_threshold: int = 3

        self._empty = True
    
    def initialize(self, start_node: Hashable | None = None, num_candidates: int | None = None, group2remaining_candidate_ids: List[Set[int]] | None = None) -> None:


        if start_node is not None:
            self._add_new_node(start_node, num_candidates, group2remaining_candidate_ids=group2remaining_candidate_ids)

        if self._verbose_level >= 1:
            _log(f"\nGraph is initialized with node: {self._nodes[start_node]}")
            self.dump()

    def record_test(
        self,
        node: Hashable,
        edge_idx: Hashable,
        success: bool,
        target_node: Optional[Hashable] = None,
        target_num_candidates: Optional[int] = None,
        group2remaining_candidate_ids: Optional[List[Set[int]]] = None,
        suspicious_transition: bool = False,
    ) -> None:

        if node not in self._nodes:
            raise KeyError(f"unknown node {node!r}") # TODO: alternatively, add it to the graph
        node_info = self._nodes[node]

        if node_info.closed:
            if target_node == self._nodes[node].edge_data["target"][edge_idx]:
                if self._verbose_level >= 1:
                    _log(f"Node {node!r} is closed, skipping test {edge_idx!r}")
                return
            else:
                if self._verbose_level >= 1:
                    _log(f"Node {node!r} is closed, we perform the test only if the target node is closer to frontier than the original target node. It will allow to fix the broken transition.")
                dist_to_frontier = self._dist.get(target_node, 0) # 0 if it wasn't previously recorded (so it's in the frontier)
                prev_target_node = self._nodes[node].edge_data["target"][edge_idx]
                prev_dist_to_frontier = self._dist.get(prev_target_node, INFINITY)

                if dist_to_frontier < prev_dist_to_frontier:
                    if self._verbose_level >= 1:
                        _log(f"Target node {target_node!r} is closer to frontier than the original target node {prev_target_node!r}, we perform the test")
                else:
                    if self._verbose_level >= 1:
                        _log(f"Target node {target_node!r} is further from frontier than the original target node {prev_target_node!r}, we skip the test")
                    return

        # store metadata immediately
        if self._verbose_level >= 1:
            _log(f"Recording action {edge_idx} from {node} to {target_node} with success {success}")

        if suspicious_transition:
            self.suspicious_transitions[(node, edge_idx, target_node)] = self.suspicious_transitions.get((node, edge_idx, target_node), 0) + 1
            _log(f"Suspicious transition detected: {node, edge_idx, target_node}, count: {self.suspicious_transitions[(node, edge_idx, target_node)]}")

            if self.suspicious_transitions[(node, edge_idx, target_node)] < self.suspicious_transitions_threshold:
                _log(f"It will be ignored for now, but will be allowed after {self.suspicious_transitions_threshold} attempts")
                return
            else:
                _log(f"Transition is recorded as permanent")
        
        node_info.record_test(edge_idx, success, target_node)
        
        # successful hop ⇒ register edge and maybe discover a brand-new node
        if success == 1:
            if target_node is None:
                raise ValueError("target_node required when success=True")

            if target_node not in self._nodes:
                new_node = True
                if target_num_candidates is None:
                    raise ValueError(
                        "target_num_candidates required for a new node"
                    )
                self._add_new_node(target_node, target_num_candidates, group2remaining_candidate_ids=group2remaining_candidate_ids)
            else:
                new_node = False


            self._G[node].add((edge_idx, target_node))
            self._G_rev[target_node].add((edge_idx, node))

            if not self._nodes[node].has_open_group(self.active_group):
                self._close_node(node)

            if self._nodes[target_node].has_open_group(self.active_group):
                # self._tighten_from_new_source(target_node)
                self._rebuild_distances()
            else:
                self._close_node(target_node)
                self._maybe_advance_group(target_node)

        else:
            if not self._nodes[node].has_open_group(self.active_group):
                self._close_node(node)
                self._maybe_advance_group(node)

        if self._verbose_level >= 1:
            if success == 1:
                success_str = "succeeded"
            elif success == -1:
                success_str = "threw an error"
            else:
                success_str = "failed"

            _log(f"\n\nNode {node!r} candidate {edge_idx!r} {success_str}:")
            _log(f"Source node:\n{self._nodes[node]}")
            if success == 1:
                _log(f"{'NEW' if new_node else 'Existing'} target node:\n{self._nodes[target_node]}")
        self.dump()

    def get_distance(self, node: Hashable) -> Optional[int]:
        d = self._dist.get(node)
        return None if d is None or d == float("inf") else d

    def get_next_hop(self, node: Hashable) -> Optional[Hashable]:
        # NOTE: DEPRECATED
        # Return the node itself only if it truly has open edges in the active group
        if node in self._frontier: # and self._nodes[node].has_open_group(self.active_group):
            return node
        nxt = self._next.get(node)
        if nxt is None:
            return None
        # _next may store (edge_idx, next_node); return the node only
        if isinstance(nxt, tuple) and len(nxt) == 2:
            return nxt[1]
        return nxt

    def edge_info(self, node: Hashable, edge_idx: Hashable) -> np.ndarray:
        return self._nodes[node].edge_data[edge_idx]

    def is_finished(self) -> bool:
        return not self._frontier

    @property
    def active_group(self) -> int:
        return self._active_group
    
    @property
    def empty(self) -> bool:
        return self._empty

    def _add_new_node(self, node: Hashable, 
        n_candidates: int, 
        group2remaining_candidate_ids: Optional[List[Set[int]]] = None
        ) -> None:

        if n_candidates < 1:
            raise ValueError("num_candidates must be positive")

        self._nodes[node] = NodeInfo(node, n_candidates, self._n_groups, group2remaining_candidate_ids=group2remaining_candidate_ids)
        self._G[node] = set()
        self._G_rev[node] = set()

        if self._empty:
            self._empty = False

        if self._nodes[node].has_open_group(self.active_group):
            self._frontier.add(node)
        else:
            self._close_node(node)
            self._maybe_advance_group(node)


    def _close_node(self, node: Hashable) -> None:
        node_info = self._nodes[node]
        if node_info.closed:
            return
        node_info.closed = True
        self._frontier.discard(node)
        self._rebuild_distances() # removal from frontier may increase some distances in the graph

    def _tighten_from_new_source(self, src: Hashable) -> None:
        # NOTE: is not used anymore
        dq = deque([src])
        self._dist[src] = 0
        self._nodes[src].distance = 0
        while dq:
            v = dq.popleft()
            v_dist = self._dist.get(v, INFINITY)
            for edge_idx, u in self._G_rev.get(v, ()):  # (edge_idx, source_node)
                initial_u_dist = self._dist.get(u, INFINITY)
                u_edge_data = self._nodes[u].edge_data
                u_edge_data["distance"][edge_idx] = self._nodes[v].distance + 1
                updated_u_dist = u_edge_data["distance"][u_edge_data["group"] <= self.active_group].min()
                self._nodes[u].distance = updated_u_dist
                self._dist[u] = updated_u_dist
                if updated_u_dist > initial_u_dist:
                    dq.append(u)

    def _rebuild_distances(self) -> None:
        """
        Rebuild the distances from the frontier nodes in the graph.
        """
        self._dist.clear()
        self._next.clear()
        dq = deque(self._frontier)
        for node, node_info in self._nodes.items():
            node_info.distance = INFINITY
            self._dist[node] = INFINITY
        for src in self._frontier:
            self._nodes[src].distance = 0
            self._dist[src] = 0
        while dq:
            v = dq.popleft()
            v_dist = self._dist.get(v, INFINITY)
            for edge_idx, u in self._G_rev.get(v, ()):  # (edge_idx, source_node)
                u_info = self._nodes[u]
                u_dist = self._dist.get(u, INFINITY)
                u_info.edge_data["distance"][edge_idx] = v_dist + 1
                if u_dist > u_info.edge_data["distance"][edge_idx]:
                    u_info.distance = u_info.edge_data["distance"][edge_idx]
                    self._dist[u] = u_info.edge_data["distance"][edge_idx]
                    self._next[u] = (edge_idx, v)
                    dq.append(u)

    def _maybe_advance_group(self, current_node: Hashable) -> None:
        """
        If it's not possible to reach any frontier node from the current node,
        given the current active group, advance to the next higher group id and rebuild distances.
        """

        distance = self._nodes[current_node].distance
        while distance == INFINITY and self.active_group < self._n_groups - 1:
            _log(f"Node {current_node!r} is not reachable from any frontier node under {self.active_group}, advancing to the next group")

            self._active_group += 1
            self._dist.clear()
            self._next.clear()
            self._frontier.clear()

            for node, node_info in self._nodes.items():
                node_info.active_group = self.active_group
                if node_info.has_open_group(self.active_group):
                    self._frontier.add(node)
                    node_info.closed = False

            self._rebuild_distances()
            distance = self._dist.get(current_node)
        
    def dump(self) -> None:
        if self._verbose_level >= 1:
            _log("=== explorer state ===")
            _log("frontier :", self._frontier)
            _log("N nodes  :", len(self._nodes))
            _log("N edged candidates  :", sum(len(node_info.edge_data) for node_info in self._nodes.values()))
            if self._verbose_level >= 2:
                _log("Graph    :", self._G)
                _log("dist     :", self._dist)
                _log("next hop :", self._next)
            _log("======================")

    def print_all_nodes(self) -> None:
        for node_info in self._nodes.values():
            _log(node_info)

    def choose_edge(self, node: Hashable, return_reasoning: bool = False) -> Hashable:
        # TODO: make it possible to choose completely random edge
        node_info = self._nodes[node]
        if node_info.has_open_group(self.active_group):
            untested_edges = []
            for group_id in range(self.active_group + 1):
                untested_edges.extend(node_info.group2remaining_candidate_ids[group_id])
            if not untested_edges:
                raise ValueError("No untested edges in the current group while the group is open")

            edge_idx = random.choice(untested_edges)
            reasoning = f"Randomly chose untested edge {edge_idx} from group {self.active_group} with {node_info.group2remaining_candidate_ids} group2candidates\n"
        else:
            lowest_dist = node_info.distance
            _log(f"Lowest dist: {lowest_dist}")
            # _log(f"Node info: {node_info}")
            edges_with_lowest_dist = [edge_idx for edge_idx, edge_data in enumerate(node_info.edge_data) if edge_data["distance"] <= lowest_dist and edge_data["result"] == 1 and edge_data["group"] <= self.active_group]
            edge_idx = random.choice(edges_with_lowest_dist)
            reasoning = f"Chose edge {edge_idx} with lowest dist {lowest_dist}\n"

        if return_reasoning:  # [port] repr formats the whole edge table; skip it when unused
            reasoning += f"Node info: {node_info}\n"
        

        if return_reasoning:
            return edge_idx, reasoning
        else:
            return edge_idx



# ---------------------------------------------------------------------------
# 2. FrameProcessor (MIT, Evgenii Rudakov)
# ---------------------------------------------------------------------------

class FrameProcessor:
    OFFSETS4: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
    OFFSETS8: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self):
        self.connectivity_rank = 4
        self.status_bar_mode = "rule"
        self.status_bar_distance_threshold = 3
        self.status_bar_ratio_threshold = 5
        self.status_bar_twins_threshold = 3
        self.frame_shape = (64, 64)

        self.status_bar_color = 16
        self.minimal_width = 2
        self.maximal_width = 32
        self.non_salient_color = set([0,1,2,3,4,5])
        self.salient_color = set([6,7,8,9,10,11,12,13,14,15])

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        pass

    def segment_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Segment `frame` into {self.connectivity_rank}-connected components (same color).

        NOTE: the twins identification increases complexity of the algorithm to O(n^2)

        Returns
        -------
        list[dict]
            One dict per component with keys
            - bounding_box : (x1, y1, x2, y2)   # inclusive pixel coords
            - color        : int                # original greyscale value
            - area         : int                # pixel count
            - is_rectangle : bool               # fully fills its bounding box
            - number_of_twins : int             # number of other components considered twins
            - twin_ids     : list[int]          # ids (1-based) of those twins
                NOTE: here we don't check shapes of the twins thoroughly

        """

        h, w = frame.shape
        label_map = np.zeros((h, w), dtype=int) - 1 # -1 = unvisited
        components: list[dict] = []
        cid = -1                                          # component id counter

        offsets = self.OFFSETS4 if self.connectivity_rank == 4 else self.OFFSETS8

        # --- first pass: flood-fill each blob ---------------------------------
        for y in range(h):
            for x in range(w):
                if label_map[y, x] != -1:                      # already labelled
                    continue
                cid += 1
                color = int(frame[y, x])
                q = deque([(y, x)])
                label_map[y, x] = cid

                min_x = max_x = x
                min_y = max_y = y
                area = 0

                while q:                                 # BFS
                    cy, cx = q.popleft()
                    area += 1
                    min_x, max_x = min(min_x, cx), max(max_x, cx)
                    min_y, max_y = min(min_y, cy), max(max_y, cy)

                    for dy, dx in offsets:
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < h and 0 <= nx < w
                            and label_map[ny, nx] == -1 # not visited
                            and frame[ny, nx] == color
                        ):
                            label_map[ny, nx] = cid
                            q.append((ny, nx))

                # rectangle test
                rect_area = (max_x - min_x + 1) * (max_y - min_y + 1)
                is_rect = area == rect_area

                components.append(
                    dict(
                        bounding_box=(min_x, min_y, max_x, max_y),
                        color=color,
                        area=area,
                        is_rectangle=is_rect,
                    )
                )

        # --- second pass: identify twins --------------------------------------
        # here: simple rule → same area, same rectangle status, and same color
        for i, comp in enumerate(components):
            twins = [
                j
                for j, other in enumerate(components)
                if i != j # skip self
                and other["area"] == comp["area"]
                and other["is_rectangle"] == comp["is_rectangle"]
                and other["color"] == comp["color"]
            ]
            comp["number_of_twins"] = len(twins)
            comp["twin_ids"] = twins

        return label_map, components

    def identify_status_bars(self, segmented_frame: np.ndarray, frame_segments: list[dict]) -> tuple[list[list[dict]] | None, np.ndarray]:
        """
        Identify the status bars from the frame segments
        Return a list of dictionaries and a frame mask.
        The list of dictionaries is the same as the input list of dictionaries in frame_segments, but with "id" key added.
        The frame mask is a binary mask where the status bars are 1 and the rest are 0.
        """
        if self.status_bar_mode == "crude":
            status_bar_mask = self.identify_status_bars_crude()
            status_bar_segments_list = None
        elif self.status_bar_mode == "rule" or self.status_bar_mode == "move":
            status_bar_segments_list, status_bar_mask = self.identify_status_bars_with_rule(segmented_frame, frame_segments)
            if self.status_bar_mode == "move":
                raise NotImplementedError("'move' mode is not implemented yet")
        else:
            raise ValueError(f"Invalid status bar mode: {self.status_bar_mode}")
        return status_bar_segments_list, status_bar_mask

    def identify_status_bars_crude(self) -> np.ndarray:
        status_bar_mask = np.zeros(self.frame_shape)
        status_bar_mask[:self.status_bar_distance_threshold, :] = 1
        status_bar_mask[-self.status_bar_distance_threshold:, :] = 1
        status_bar_mask[:, :self.status_bar_distance_threshold] = 1
        status_bar_mask[:, -self.status_bar_distance_threshold:] = 1
        return status_bar_mask
       
    def identify_status_bars_with_rule(self, segmented_frame: np.ndarray, frame_segments: list[dict]) -> tuple[list[list[dict]], np.ndarray]:
        """
        Identify the status bars from the frame segments
        Return a list of dictionaries and a frame mask.
        The list of dictionaries is the same as the input list of dictionaries in frame_segments, but with "id" key added.
        The frame mask is a binary mask where the status bars are 1 and the rest are 0.
        """

        # modes:
            # crude: remove all screen edges 
            # rule: rule-based
            # move: rule-based + movement after the first action 


        # the rules are:
            # the status bars are close to the edges of the screen
            # they can be in any orientation
            # the can be duplicated from both sides of the screen
            # there are 2 types of status bars:
                # 1. the line 
                # 2. the dots, for the dots there should be at least 3 twins


        checked_segment_ids = set()
        status_bar_segment_ids_list = [] # list[list[int]]
        for i, segment in enumerate(frame_segments):

            status_bar_segment_ids = [i]

            if i in checked_segment_ids:
                continue
            checked_segment_ids.add(i)
            on_edge_list = self.check_segment_fully_on_edge(segment, edges=['any'])
            if len(on_edge_list) == 0:
                continue
            directions = []
            if 'left' in on_edge_list or 'right' in on_edge_list:
                directions.append('vertical')
            if 'top' in on_edge_list or 'bottom' in on_edge_list:
                directions.append('horizontal')
            if len(directions) == 2:
                direction = 'any'
            else:
                direction = directions[0]
            is_long_ratio = self.check_segment_ratio(segment, direction=direction)  

            if not is_long_ratio:
                twin_ids_on_edge_list = self.segment_twins_on_edge(segment, frame_segments)
                for twin_id in twin_ids_on_edge_list:
                    checked_segment_ids.add(twin_id)
                if len(twin_ids_on_edge_list) + 1 < self.status_bar_twins_threshold:
                    continue
                status_bar_segment_ids.extend(twin_ids_on_edge_list)

            status_bar_segment_ids_list.append(status_bar_segment_ids)

        status_bar_segments_list = []
        status_bar_mask = np.zeros(segmented_frame.shape, dtype=bool)

        for i, status_bar_segment_ids in enumerate(status_bar_segment_ids_list):
            status_bar_segments = []
            for status_bar_segment_id in status_bar_segment_ids:
                status_bar_mask[segmented_frame == status_bar_segment_id] = 1

                status_bar_segments.append(frame_segments[status_bar_segment_id])
            status_bar_segments_list.append(status_bar_segments)

        return status_bar_segments_list, status_bar_mask

    def check_segment_fully_on_edge(self, segment: dict, edges: list[str] | None = None) -> list[str]:
        """
        Check if the segment is fully on the edge of the screen
        """
        x1, y1, x2, y2 = segment["bounding_box"]
        if edges is None:
            edges = ['any']
        for edge in edges:
            assert edge in ['any', 'left', 'right', 'top', 'bottom']

        result = []

        if 'left' in edges or 'any' in edges:
            max_x = max(x1, x2)
            if max_x < self.status_bar_distance_threshold:
                result.append('left')
        if 'right' in edges or 'any' in edges:
            min_x = min(x1, x2)
            if min_x > self.frame_shape[1] - self.status_bar_distance_threshold:
                result.append('right')
        if 'top' in edges or 'any' in edges:
            max_y = max(y1, y2)
            if max_y < self.status_bar_distance_threshold:
                result.append('top')
        if 'bottom' in edges or 'any' in edges:
            min_y = min(y1, y2)
            if min_y > self.frame_shape[0] - self.status_bar_distance_threshold:
                result.append('bottom')
        # NOTE: there can be some mess with the y-axis direction (should it start from the top or the bottom), need to double check
        return result

    def check_segment_ratio(self, segment: dict, direction: str | None = None) -> bool:
        """
        Check if the segment is a status bar
        """
        if direction is None:
            direction = 'any'
        assert direction in ['any', 'horizontal', 'vertical']

        x_length, y_length = segment["bounding_box"][2] - segment["bounding_box"][0] + 1, segment["bounding_box"][3] - segment["bounding_box"][1] + 1
        x_to_y_ratio = x_length / y_length
        if x_to_y_ratio >= self.status_bar_ratio_threshold and direction in ('any', 'horizontal'):
            return True
        if x_to_y_ratio <= 1 / self.status_bar_ratio_threshold and direction in ('any', 'vertical'):
            return True
        return False

    def segment_twins_on_edge(self, segment: dict, frame_segments: list[dict], edges: list[str] | None = None) -> list[int]:
        """
        Check if the segment has twins on the same edge
        """

        if edges is None:
            edges = self.check_segment_fully_on_edge(segment, edges=['any'])
            if len(edges) == 0:
                return []

        twins = []
        for twin_id in segment["twin_ids"]:
            twin = frame_segments[twin_id]
            twin_edges = self.check_segment_fully_on_edge(twin, edges=edges)
            if len(twin_edges) > 0:
                twins.append(twin_id)
        
        return twins
        
    
    def hash_frame(self, frame: np.ndarray) -> str:
        """
        Deterministic 128-bit hash for an integer-valued NumPy array whose
        elements are in the range 0 … 15 (4 bits).

        • Compact: packs two elements per byte before hashing  
        • Stable: identical digest across Python versions & interpreter restarts  
        • Shape-aware: (m, n) and (n, m) views do NOT collide  
        • Dependency-free: only stdlib hashlib
        """
        # TODO: maybe just convert a matrix to a number and store it
        frame = np.asarray(frame, dtype=np.uint8, order='C')

        # ---- pack two 4-bit values into each byte ---------------------------
        flat = frame.ravel()
        if flat.size & 1:                       # pad to even length
            flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
        packed = (flat[0::2] << 4) | (flat[1::2] & 0x0F)
        payload = packed.tobytes()

        # ---- hash with Blake2B (128-bit digest) -----------------------------
        shape_tag = frame.shape.__repr__().encode()
        return hashlib.blake2b(payload,
                            digest_size=16,   # 128 bits
                            person=shape_tag  # embeds the shape
                            ).hexdigest()


    def frame_segments_to_action_groups(self, frame_segments: list[dict], n_groups: int) -> list[list[int]]:
        """
        Assign actions to groups
        """
        group_0_segments = set()
        group_1_segments = set()
        group_2_segments = set()
        group_3_segments = set()
        group_4_segments = set()

        for segment_id, segment in enumerate(frame_segments):
            x_width, y_width = segment["bounding_box"][2] - segment["bounding_box"][0] + 1, segment["bounding_box"][3] - segment["bounding_box"][1] + 1
            is_salient = segment["color"] in self.salient_color
            is_medium_width = self.minimal_width <= x_width <= self.maximal_width and self.minimal_width <= y_width <= self.maximal_width
            is_status_bar = segment["color"] == self.status_bar_color

            assert n_groups == 5, "Only 5 groups are supported for now"

            if is_salient and is_medium_width:
                group_0_segments.add(segment_id)
            elif is_medium_width:
                group_1_segments.add(segment_id)
            elif is_salient:
                group_2_segments.add(segment_id)
            elif not is_status_bar:
                group_3_segments.add(segment_id)
            else:
                group_4_segments.add(segment_id)

        groups2segments = [group_0_segments, group_1_segments, group_2_segments, group_3_segments, group_4_segments]
        # groups2segments = groups2segments[::-1] # NOTE: temporary to check the robustness 

        return groups2segments


# ---------------------------------------------------------------------------
# 3. MyAgent (MIT-0)
# ---------------------------------------------------------------------------

SIMPLE_ACTIONS = {
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
    5: GameAction.ACTION5,
}


def shortest_path_length(graph: Dict[Hashable, Set[Tuple[int, Hashable]]], start: Hashable,
                         goal: Hashable) -> Optional[int]:
    """BFS hop count from start to goal over the explorer's successful edges."""
    if start == goal:
        return 0
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        for _, nxt in graph.get(node, ()):
            if nxt == goal:
                return dist + 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    return None


class MyAgent(Agent):
    """Frame-graph explorer (just-explore) on the 2026 Agent interface."""

    # Kaggle: 9 h notebook limit, all games run concurrently in one process (Swarm threads),
    # so the per-game clock is effectively the whole run. Extra actions never lower the score
    # of solved levels, so time is the only real cap. (Agreed with Elias 2026-10-21.)
    MAX_ACTIONS: int = 10_000_000   # play_local.py / run_local.py lower this
    TIME_BUDGET_S: float = float(os.getenv("ARC3_TIME_BUDGET_S", 6.5 * 3600))  # env only for local sims
    RECORD: bool = False            # Swarm passes record=True; see start_recording()
    N_GROUPS: int = 5
    SEED: Optional[int] = None      # None -> time-based, like the original
    VARIANT: str = "just-explore-port-v2"  # v1 = RECOVERY_FIX only; v2 adds ORPHAN_NODE_FIX
    # Upstream's except-branch sets level_up=True, but main() flips it back to False on the
    # very next frame (status_bar_mask is still set), and last_hashed_frame/last_action stay
    # stale, so the same failing record_test repeats forever. True = rebuild the level graph
    # as upstream intended; False = reproduce upstream behaviour exactly.
    RECOVERY_FIX: bool = True
    # After GAME_OVER -> RESET the start frame can hash differently; that transition is flagged
    # suspicious and ignored (<3 times), so the current frame never enters the graph and
    # choose_edge() raises KeyError, which wipes the level graph (r11l: 174 wipes in 2000 steps).
    # True = add the frame as a node without the suspicious edge; False = upstream behaviour.
    ORPHAN_NODE_FIX: bool = True
    # Click candidates for ACTION6 (k_eff ablation, docs/novelty.md item 3):
    #   "segments": one candidate per connected component, in 5 priority groups (just-explore);
    #   "grid8":    the centres of an 8x8 grid of 8-pixel blocks, 64 candidates, one group;
    #   "pixels":   every pixel, 4096 candidates, one group.
    CLICK_MODE: str = "segments"
    # Priority order of click candidates (D1, docs/d1-plan.md):
    #   "default": groups for segments, flat for grid8/pixels (the stage-3 behaviour);
    #   "groups":  just-explore's 5 groups; a grid/pixel candidate takes the group of its segment;
    #   "flat":    every click candidate in group 0, so untested candidates are drawn uniformly.
    CLICK_ORDER: str = "default"
    BUDGET_BUCKETS: int = 0  # D3: 0 preserves the original state hash and policy

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if type(self.BUDGET_BUCKETS) is not int or self.BUDGET_BUCKETS < 0:
            raise ValueError("BUDGET_BUCKETS must be a nonnegative integer")
        if self.SEED is None:
            self.seed = (time.time_ns() + hash(self.game_id)) % (2**32)
        else:
            self.seed = int(self.SEED)
        random.seed(self.seed)

        self.frame_processor = FrameProcessor()
        self.graph_explorer = GraphExplorer(verbose_level=0, n_groups=self.N_GROUPS)
        self.status_bar_mask: Optional[np.ndarray] = None
        self.hashed_frame2action_results: Dict[str, np.ndarray] = {}
        self.hashed_frame2transitions: Dict[str, List[Any]] = {}
        self.last_hashed_frame: Optional[str] = None
        self.last_action: Optional[int] = None
        self.last_action_object: GameAction = GameAction.RESET
        self.level_first_frame: Optional[str] = None
        self.failed = False
        self.level_up = True
        self.last_transition_suspicious = False

        self.stop_reason = ""
        self.errors = 0
        self.level_records: List[Dict[str, Any]] = []
        self._last_xy: Optional[Tuple[int, int]] = None  # pixel of the last click, for conflict bookkeeping
        self._skip_note = False  # the next observed transition is a post-GAME_OVER RESET
        self._level = self._new_level(0)

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def start_recording(self) -> None:
        # Swarm always passes record=True, which writes every frame (~10-20 KB) to disk:
        # 25 games x ~100k actions would fill Kaggle's working directory mid-run.
        if self.RECORD:
            super().start_recording()

    KEEP_FRAMES: int = 4

    def append_frame(self, frame: FrameData) -> None:
        # The framework keeps every frame (~100 KB each as Python lists); with 25 games in
        # parallel threads on Kaggle that exhausts memory. Only frames[-1] is ever read.
        super().append_frame(frame)
        if len(self.frames) > 2 * self.KEEP_FRAMES:
            del self.frames[:-self.KEEP_FRAMES]

    # -- per-level statistics ------------------------------------------------

    def _new_level(self, index: int) -> Dict[str, Any]:
        return {"level_index": index, "actions_agent": 0, "resets": 0,
                "t0": time.perf_counter(), "start_node": None,
                # D1 diagnostics; kept across graph rebuilds within the level
                "states": {},   # hash -> {"cand_seg", "n_click", "n_seg", "obj_segs", "tests"}
                "pairs": {},    # (hash, action id) -> (last observed outcome, click point)
                "tested_in_graph": {},  # (hash, action id) -> graph epoch of the last observation
                "graph_epoch": 0,       # +1 on every explorer rebuild
                "conflicts": 0, "conflicts_same_point": 0, "retests": 0, "suspicious": 0,
                "error_kinds": defaultdict(int), "game_overs": 0,
                "budget_reference": None, "budget_bucket_values_seen": set()}

    def _diagnostics(self) -> Dict[str, Any]:
        lv = self._level
        tests = eff = repeat = redundant = states_clicked = states_no_effect = 0
        ranks, click_classes, cov_seg, cov_obj = [], [], [], []
        for st in lv["states"].values():
            cand_seg = st["cand_seg"]
            if st["n_click"]:
                cov_seg.append(len(set(cand_seg.tolist())) / st["n_seg"])
                if st["obj_segs"]:
                    cov_obj.append(len(st["obj_segs"] & set(cand_seg.tolist())) / len(st["obj_segs"]))
            clicks = [(a, o) for a, o in st["tests"] if a < st["n_click"]]
            if not clicks:
                continue
            states_clicked += 1
            tests += len(clicks)
            eff += sum(1 for _, o in clicks if o)
            repeat += len(clicks) - len({int(cand_seg[a]) for a, _ in clicks})
            classes = len({o for _, o in clicks})
            click_classes.append(classes)
            redundant += len(clicks) - classes
            rank = next((i + 1 for i, (_, o) in enumerate(clicks) if o), None)
            if rank is None:
                states_no_effect += 1
            else:
                ranks.append(rank)
        ex = self.graph_explorer
        return {
            "click_tests": tests,
            "click_effective": eff,
            "click_hit_rate": eff / tests if tests else None,
            "first_effect_rank_mean": float(np.mean(ranks)) if ranks else None,
            "states_clicked": states_clicked,
            "states_no_effect": states_no_effect,
            "seg_repeat_tests": repeat,
            "click_effect_classes_mean": float(np.mean(click_classes)) if click_classes else None,
            "redundant_click_tests": redundant,
            "coverage_seg_mean": float(np.mean(cov_seg)) if cov_seg else None,
            "coverage_obj_mean": float(np.mean(cov_obj)) if cov_obj else None,
            "conflicts": lv["conflicts"],
            "conflicts_same_point": lv["conflicts_same_point"],
            "retests": lv["retests"],
            "suspicious": lv["suspicious"],
            "game_overs": lv["game_overs"],
            "errors": sum(lv["error_kinds"].values()),
            "error_kinds": dict(lv["error_kinds"]),
            "frontier_empty": bool(not ex.empty and ex.is_finished()),
            "active_group": ex.active_group,
            "click_mode": self.CLICK_MODE,
            "click_order": self.CLICK_ORDER,
            **({"budget_buckets": self.BUDGET_BUCKETS,
                "budget_bars_detected": len(lv["budget_reference"] or []),
                "budget_bucket_values_seen": sorted(lv["budget_bucket_values_seen"])}
               if self.BUDGET_BUCKETS else {}),
        }

    def _note_state(self, hashed_frame: str, segmented_frame: np.ndarray, frame_segments: list,
                    seg_groups: List[Set[int]], click_points: Optional[List[Tuple[int, int]]],
                    num_click_actions: int) -> None:
        states = self._level["states"]
        if hashed_frame in states:
            return
        if click_points is None:  # segments: candidate i is segment i
            cand_seg = np.arange(num_click_actions, dtype=np.int32)
        else:
            cand_seg = np.array([segmented_frame[y, x] for x, y in click_points], dtype=np.int32)
        states[hashed_frame] = {"cand_seg": cand_seg, "n_click": num_click_actions,
                                "n_seg": max(1, len(frame_segments)),
                                "obj_segs": set(seg_groups[0]) | set(seg_groups[1]), "tests": []}

    def _note_outcome(self, state: str, action: int, outcome: str, suspicious: bool = False) -> None:
        """Record what (state, action id) did. outcome: target hash, "" = no visible change,
        "GAME_OVER" or "LEVEL_UP". A conflict is an observation that differs from the previous
        observation of the same pair; it counts as same-point when the click landed on the same
        pixel both times (segments pick a random pixel of the segment on every click)."""
        lv = self._level
        lv["suspicious"] += int(suspicious)
        key = (state, action)
        point = self._last_xy
        prev = lv["pairs"].get(key)
        if prev is None:
            if state in lv["states"]:
                lv["states"][state]["tests"].append((action, outcome))
        else:
            if lv["tested_in_graph"][key] != lv["graph_epoch"]:
                lv["retests"] += 1  # the graph was rebuilt and the pair is tested again
            if prev[0] != outcome:
                lv["conflicts"] += 1
                lv["conflicts_same_point"] += int(prev[1] == point)
        lv["pairs"][key] = (outcome, point)
        lv["tested_in_graph"][key] = lv["graph_epoch"]

    @staticmethod
    def _error_kind(exc: Exception) -> str:
        """Group choose_action exceptions; only outcome_flip signals a same-pair different-result event."""
        msg = str(exc)
        if msg.startswith("Old value"):
            return "outcome_flip"        # same (state, action) moved one time and stayed another
        if "must be untested" in msg:
            return "explorer_assert"     # upstream re-records an already tested edge
        if "No available actions" in msg or "No untested edges" in msg:
            return "exhausted"
        if isinstance(exc, KeyError):
            return "unknown_node"
        return type(exc).__name__

    def _pending_outcome(self, outcome: str) -> None:
        """Settle the last action's outcome when no further choose_action will observe it
        (level up, game over)."""
        if self.last_hashed_frame is not None and self.last_action is not None:
            self._note_outcome(self.last_hashed_frame, self.last_action, outcome)

    def _close_level(self, solved: bool) -> None:
        lv = self._level
        nodes = self.graph_explorer._nodes
        candidates = [n.total_candidates for n in nodes.values()]
        tested = [int((n.edge_data["result"] != 0).sum()) for n in nodes.values()]
        # distinct outcomes among the tested candidates of each state: each different target frame
        # counts once, and "no visible change" counts once (effect classes, docs/results-stage3.md)
        effects = []
        for n in nodes.values():
            ed = n.edge_data
            succ = set(ed["target"][ed["result"] == 1].tolist())
            effects.append(len(succ) + int((ed["result"] == -1).any()))
        depth = None
        if solved and lv["start_node"] is not None and self.last_hashed_frame is not None:
            hops = shortest_path_length(self.graph_explorer._G, lv["start_node"], self.last_hashed_frame)
            depth = None if hops is None else hops + 1  # +1 for the winning action itself
        self.level_records.append({
            "level": lv["level_index"] + 1,
            "solved": solved,
            "actions_agent": lv["actions_agent"],
            "resets": lv["resets"],
            "states_discovered_Ng": len(nodes),
            "k_eff": float(np.mean(candidates)) if candidates else None,
            "k_tested_mean": float(np.mean(tested)) if tested else None,
            "tested_pairs": int(sum(tested)),
            "effect_classes_mean": float(np.mean(effects)) if effects else None,
            "goal_depth_D": depth,
            "seconds": time.perf_counter() - lv["t0"],
            "diag": self._diagnostics(),
        })

    # -- main loop -----------------------------------------------------------

    def main(self) -> None:
        """Agent.main plus level bookkeeping, error recovery and a time budget."""
        self.timer = time.time()
        started = time.perf_counter()
        prev_levels = self.frames[-1].levels_completed
        while not self.is_done(self.frames, self.frames[-1]) and self.action_counter < self.MAX_ACTIONS:
            if time.perf_counter() - started > self.TIME_BUDGET_S:
                self.stop_reason = "time_budget"
                break
            latest = self._convert_raw_frame_data(self.arc_env.observation_space if self.arc_env else None)
            try:
                action = self.choose_action(self.frames, latest)
            except Exception as exc:  # same recovery as the original: rebuild the level graph
                self.errors += 1
                self._level["error_kinds"][self._error_kind(exc)] += 1
                self.failed = True
                self.level_up = True
                logger.warning("%s: choose_action failed (%s); re-initializing the explorer", self.game_id, exc)
                action = self.last_action_object
                if self.RECOVERY_FIX:
                    self.status_bar_mask = None  # keeps level_up=True until the next choose_action
                    self.last_hashed_frame = None
                    self.last_action = None
            frame = self.take_action(action)
            self._level["actions_agent"] += 1
            if action is GameAction.RESET:
                self._level["resets"] += 1
            if frame is not None:
                if frame.levels_completed > prev_levels:
                    self._pending_outcome("LEVEL_UP")
                    self._close_level(solved=True)
                    self._level = self._new_level(frame.levels_completed)
                    self.level_up = True
                    self.status_bar_mask = None
                elif self.status_bar_mask is not None:
                    self.level_up = False
                prev_levels = frame.levels_completed
                self.append_frame(frame)
            self.action_counter += 1
        if not self.stop_reason:
            self.stop_reason = "win" if self.frames[-1].state is GameState.WIN else "max_actions"
        if self.stop_reason != "win":
            self._close_level(solved=False)
        self.cleanup()

    # -- policy (HeuristicAgent.choose_action, minus sleep and plotting) -----

    def _transition_data(self, hashed_frame: str, num_actions: int) -> np.ndarray:
        if hashed_frame not in self.hashed_frame2action_results:
            self.hashed_frame2action_results[hashed_frame] = np.zeros(num_actions)
            self.hashed_frame2transitions[hashed_frame] = [0] * num_actions
        return self.hashed_frame2action_results[hashed_frame]

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state is GameState.NOT_PLAYED:
            self.last_hashed_frame = None
            self.last_action = None
            if self.failed:
                self.level_up = True
                self.failed = False
            return GameAction.RESET
        if latest_frame.state is GameState.GAME_OVER:
            if not self._skip_note:
                self._level["game_overs"] += 1
                self._pending_outcome("GAME_OVER")
                self._skip_note = True
            self.last_transition_suspicious = True
            return GameAction.RESET

        grids = np.array(latest_frame.frame, dtype=np.uint8)
        if grids.size == 0:  # no frame to look at: random non-reset action, as in the original
            action = random.choice([a for a in GameAction if a is not GameAction.RESET])
            if action.is_complex():
                action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
            return action
        num_frames = grids.shape[0]
        frame = grids[-1]  # several grids = animation; the last one is the settled state

        if self.level_up:
            seg, segs = self.frame_processor.segment_frame(frame)
            bars, self.status_bar_mask = self.frame_processor.identify_status_bars(seg, segs)
            if self.BUDGET_BUCKETS and self._level["budget_reference"] is None:
                # Freeze first-frame bar groups for this level, including across recovery.
                self._level["budget_reference"] = [
                    (np.isin(seg, [i for i, item in enumerate(segs)
                                   if any(item is bar for bar in group)]), group[0]["color"])
                    for group in (bars or [])
                ]
            self.hashed_frame2action_results = {}
            self.hashed_frame2transitions = {}

        if self.BUDGET_BUCKETS:
            fractions = [float(np.count_nonzero((frame == color) & mask & self.status_bar_mask))
                         / int(mask.sum())
                         for mask, color in self._level["budget_reference"]]
            bucket = min(self.BUDGET_BUCKETS - 1, int(min(fractions, default=1.0) * self.BUDGET_BUCKETS))
            self._level["budget_bucket_values_seen"].add(bucket)

        frame[self.status_bar_mask] = 16
        segmented_frame, frame_segments = self.frame_processor.segment_frame(frame)
        available = latest_frame.available_actions

        num_actions = 0
        num_click_actions = 0
        arrow_actions: List[GameAction] = []
        click_points: Optional[List[Tuple[int, int]]] = None
        seg_groups = self.frame_processor.frame_segments_to_action_groups(frame_segments, n_groups=self.N_GROUPS)
        order = self.CLICK_ORDER
        if order == "default":
            order = "groups" if self.CLICK_MODE == "segments" else "flat"
        if order not in ("groups", "flat"):
            raise ValueError(f"unknown CLICK_ORDER {self.CLICK_ORDER!r}")
        if 6 in available and self.CLICK_MODE == "segments":
            num_actions += len(frame_segments)
            num_click_actions += len(frame_segments)
            if order == "groups":
                action_groups = [set(g) for g in seg_groups]
            else:
                action_groups = [set(range(num_actions))] + [set() for _ in range(self.N_GROUPS - 1)]
        elif 6 in available:
            h, w = frame.shape
            step = 8 if self.CLICK_MODE == "grid8" else 1
            if self.CLICK_MODE not in ("grid8", "pixels"):
                raise ValueError(f"unknown CLICK_MODE {self.CLICK_MODE!r}")
            click_points = [(x, y) for y in range(step // 2, h, step) for x in range(step // 2, w, step)]
            num_actions = num_click_actions = len(click_points)
            if order == "groups":
                group_of = {s: g for g, segs in enumerate(seg_groups) for s in segs}
                action_groups = [set() for _ in range(self.N_GROUPS)]
                for i, (x, y) in enumerate(click_points):
                    action_groups[group_of[int(segmented_frame[y, x])]].add(i)
            else:
                action_groups = [set(range(num_actions))] + [set() for _ in range(self.N_GROUPS - 1)]
        else:
            action_groups = [set() for _ in range(self.N_GROUPS)]
        for action_id in available:
            if action_id in SIMPLE_ACTIONS:
                arrow_actions.append(SIMPLE_ACTIONS[action_id])
                action_groups[0].add(num_actions)
                num_actions += 1

        frame[frame == 16] = 0
        hashed_frame = self.frame_processor.hash_frame(frame)
        if self.BUDGET_BUCKETS and self._level["budget_reference"]:
            # Keep the graph edge target U32 width; never append to the hex digest.
            hashed_frame = hashlib.blake2b(
                f"{hashed_frame}:budget:{bucket}".encode(), digest_size=16).hexdigest()

        if self.level_up:
            self.level_first_frame = hashed_frame
            self._level["start_node"] = hashed_frame
            self.graph_explorer.reset()
            self._level["graph_epoch"] += 1
            self.graph_explorer.initialize(start_node=hashed_frame, num_candidates=num_actions,
                                           group2remaining_candidate_ids=action_groups)

        transition = False
        suspicious = False
        if self.last_hashed_frame is not None and not self.level_up:
            transition = hashed_frame != self.last_hashed_frame
            suspicious = hashed_frame == self.level_first_frame and num_frames > 1
            if self.last_transition_suspicious:
                suspicious = True
                self.last_transition_suspicious = False
            results = self.hashed_frame2action_results[self.last_hashed_frame]
            old_value = results[self.last_action]
            if self._skip_note:  # this transition is the RESET after a GAME_OVER, already settled
                self._skip_note = False
            else:
                self._note_outcome(self.last_hashed_frame, self.last_action, hashed_frame if transition else "",
                                   suspicious)
            results[self.last_action] = 1 if transition else -1
            self.hashed_frame2transitions[self.last_hashed_frame][self.last_action] = hashed_frame if transition else None
            self.graph_explorer.record_test(self.last_hashed_frame, self.last_action, transition, hashed_frame,
                                            target_num_candidates=num_actions,
                                            group2remaining_candidate_ids=action_groups,
                                            suspicious_transition=suspicious)
            if old_value != 0 and old_value != results[self.last_action]:
                raise ValueError(f"Old value {old_value} is not equal to new value {results[self.last_action]}")

        results = self._transition_data(hashed_frame, num_actions)
        self._note_state(hashed_frame, segmented_frame, frame_segments, seg_groups, click_points, num_click_actions)
        if hashed_frame not in self.graph_explorer._nodes and self.ORPHAN_NODE_FIX:
            self.graph_explorer._add_new_node(hashed_frame, num_actions,
                                              group2remaining_candidate_ids=action_groups)
        if hashed_frame not in self.graph_explorer._nodes:
            # e.g. after a recovered error, the current frame was never recorded
            self.graph_explorer.record_test(self.last_hashed_frame, self.last_action, transition, hashed_frame,
                                            target_num_candidates=num_actions,
                                            group2remaining_candidate_ids=action_groups,
                                            suspicious_transition=suspicious)
        if len(np.where(results != -1)[0]) == 0:
            raise ValueError(f"No available actions found for frame {hashed_frame}")

        action_id = self.graph_explorer.choose_edge(hashed_frame)
        if action_id < num_click_actions:
            if click_points is not None:
                x, y = click_points[action_id]
            else:
                points = np.argwhere(segmented_frame == action_id)
                y, x = points[random.randint(0, len(points) - 1)]
            action = GameAction.ACTION6
            action.set_data({"x": int(x), "y": int(y)})
            self._last_xy = (int(x), int(y))
        else:
            action = arrow_actions[action_id - num_click_actions]
            self._last_xy = None

        self.last_hashed_frame = hashed_frame
        self.last_action = action_id
        self.last_action_object = action
        return action
