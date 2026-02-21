from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Dict, List, Tuple
import time

Box = Tuple[int, int, int, int]


@dataclass
class _Track:
    track_id: int
    center: Tuple[float, float]
    last_seen: float


class FaceTracker:
    """Assigns stable session IDs to detected faces using centroid matching."""

    def __init__(self, max_distance: float = 120.0, max_age_seconds: float = 1.5) -> None:
        self.max_distance = max_distance
        self.max_age_seconds = max_age_seconds
        self._next_id = 1
        self._tracks: Dict[int, _Track] = {}

    @staticmethod
    def _center(box: Box) -> Tuple[float, float]:
        x, y, w, h = box
        return (x + w / 2.0, y + h / 2.0)

    def _prune_stale(self, now: float) -> None:
        stale_ids = [
            tid
            for tid, track in self._tracks.items()
            if now - track.last_seen > self.max_age_seconds
        ]
        for tid in stale_ids:
            del self._tracks[tid]

    def update(self, boxes: List[Box]) -> List[int]:
        """Returns one session ID per input box in the same order."""
        now = time.time()
        self._prune_stale(now)

        if not boxes:
            return []

        centers = [self._center(box) for box in boxes]
        assigned_track_ids: List[int] = [-1] * len(boxes)
        used_tracks = set()

        for i, center in enumerate(centers):
            best_tid = None
            best_dist = float("inf")

            for tid, track in self._tracks.items():
                if tid in used_tracks:
                    continue
                dist = hypot(center[0] - track.center[0], center[1] - track.center[1])
                if dist < best_dist and dist <= self.max_distance:
                    best_dist = dist
                    best_tid = tid

            if best_tid is None:
                best_tid = self._next_id
                self._next_id += 1

            self._tracks[best_tid] = _Track(track_id=best_tid, center=center, last_seen=now)
            assigned_track_ids[i] = best_tid
            used_tracks.add(best_tid)

        return assigned_track_ids
