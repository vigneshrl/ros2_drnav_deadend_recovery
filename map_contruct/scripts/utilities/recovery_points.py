from dataclasses import dataclass
from typing import List, Optional
import time
import math


@dataclass
class RecoveryPoint:
    """Stores a spatial position where the robot had at least one open direction."""
    x: float                    # Robot x in map frame
    y: float                    # Robot y in map frame
    open_directions: List[int]  # [front, left, right] — 1=open, 0=blocked
    rank: int                   # Number of open directions (1–3)
    timestamp: float            # When this point was recorded
    confidence: List[float]     # Raw path probabilities [front, left, right]


class RecoveryPointManager:
    """
    Tracks spatial positions where the robot had open paths.
    Used for dead-end recovery: when all directions are blocked,
    navigate back to the best stored recovery point.

    Threshold: 0.56 (consistent with infer_vis and cost_layer_processor)
    """

    def __init__(self,
                 confidence_threshold: float = 0.56,
                 max_stored_points: int = 50,
                 min_distance_m: float = 1.0,
                 max_age_s: float = 60.0):
        """
        Args:
            confidence_threshold: path probability above which direction is 'open'
            max_stored_points:    cap on total stored points (oldest dropped first)
            min_distance_m:       spatial deduplication radius — within this radius,
                                  only keep the point with the highest rank
            max_age_s:            expire points older than this many seconds
        """
        self.confidence_threshold = confidence_threshold
        self.max_stored_points = max_stored_points
        self.min_distance_m = min_distance_m
        self.max_age_s = max_age_s

        self.recovery_points: List[RecoveryPoint] = []

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def process_probabilities(self,
                              path_probs: List[float],
                              x: float,
                              y: float) -> Optional[RecoveryPoint]:
        """
        Main entry point for cost_layer_processor.

        Takes already-computed path probabilities [front, left, right],
        robot position in map frame, and decides whether this is a
        recovery point worth storing.

        If a nearby point already exists:
          - keep the new one if it has a higher rank (more open directions)
          - discard otherwise

        Returns the new RecoveryPoint if stored, else None.
        """
        probs = list(path_probs)[:3]
        open_dirs = [1 if p > self.confidence_threshold else 0 for p in probs]
        num_open = sum(open_dirs)

        # A recovery point requires at least one open direction
        if num_open == 0:
            return None

        # Expire stale points first
        self._expire_old_points()

        # Spatial deduplication — check for nearby existing point
        nearby_idx = self._find_nearby_index(x, y)
        if nearby_idx is not None:
            existing = self.recovery_points[nearby_idx]
            if num_open > existing.rank:
                # Replace the existing point with the higher-rank one
                point = RecoveryPoint(
                    x=x, y=y,
                    open_directions=open_dirs,
                    rank=num_open,
                    timestamp=time.time(),
                    confidence=probs
                )
                self.recovery_points[nearby_idx] = point
                return point
            # Existing point is equal or better — discard new one
            return None

        point = RecoveryPoint(
            x=x, y=y,
            open_directions=open_dirs,
            rank=num_open,
            timestamp=time.time(),
            confidence=probs
        )

        self.recovery_points.append(point)

        # Enforce max capacity — drop oldest
        if len(self.recovery_points) > self.max_stored_points:
            self.recovery_points.pop(0)

        return point

    def is_dead_end(self, path_probs: List[float]) -> bool:
        """True if all three directions are blocked (all below threshold)."""
        return all(p <= self.confidence_threshold for p in list(path_probs)[:3])

    # ------------------------------------------------------------------
    # Retrieval  (all expire stale points before returning)
    # ------------------------------------------------------------------

    def get_last_recovery_point(self) -> Optional[RecoveryPoint]:
        """Most recently stored recovery point."""
        self._expire_old_points()
        if not self.recovery_points:
            return None
        return self.recovery_points[-1]

    def get_best_recovery_point(self) -> Optional[RecoveryPoint]:
        """Recovery point with the most open directions (highest rank)."""
        self._expire_old_points()
        if not self.recovery_points:
            return None
        return max(self.recovery_points, key=lambda p: p.rank)

    def get_nearest_recovery_point(self, robot_x: float, robot_y: float) -> Optional[RecoveryPoint]:
        """Closest recovery point to the robot's current position."""
        self._expire_old_points()
        if not self.recovery_points:
            return None
        return min(self.recovery_points,
                   key=lambda p: math.hypot(p.x - robot_x, p.y - robot_y))

    def get_all_points(self) -> List[RecoveryPoint]:
        """Return all valid (non-expired) recovery points."""
        self._expire_old_points()
        return list(self.recovery_points)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_nearby_index(self, x: float, y: float) -> Optional[int]:
        """Return index of the first existing point within min_distance_m, or None."""
        for i, rp in enumerate(self.recovery_points):
            if math.hypot(rp.x - x, rp.y - y) < self.min_distance_m:
                return i
        return None

    def _expire_old_points(self):
        cutoff = time.time() - self.max_age_s
        self.recovery_points = [rp for rp in self.recovery_points
                                if rp.timestamp >= cutoff]
