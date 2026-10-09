"""Planar geometry in normalized coordinates (origin top-left, x right, y down)."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict


class Point(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float

    def distance(self, other: Point) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def scaled(self, sx: float, sy: float) -> Point:
        return Point(x=self.x * sx, y=self.y * sy)


class Rect(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    width: float
    height: float

    @property
    def max_x(self) -> float:
        return self.x + self.width

    @property
    def max_y(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height / 2)

    def contains(self, p: Point) -> bool:
        return self.x <= p.x <= self.max_x and self.y <= p.y <= self.max_y


UNIT_RECT = Rect(x=0, y=0, width=1, height=1)


class Quad(BaseModel):
    """Four named corners, as found by a pack detector."""

    model_config = ConfigDict(frozen=True)

    top_left: Point
    top_right: Point
    bottom_right: Point
    bottom_left: Point

    @property
    def corners(self) -> list[Point]:
        return [self.top_left, self.top_right, self.bottom_right, self.bottom_left]

    @classmethod
    def from_corners(cls, corners: list[Point]) -> Quad:
        return cls(top_left=corners[0], top_right=corners[1], bottom_right=corners[2], bottom_left=corners[3])

    def scaled(self, sx: float, sy: float) -> Quad:
        return Quad.from_corners([c.scaled(sx, sy) for c in self.corners])

    def rotated(self, quarter_turns: int) -> Quad:
        """Relabels corners for a pack lying rotated in the image. One quarter turn means the pack's
        top edge runs along the image's left edge (pack top-left = image bottom-left)."""
        c = self.corners
        for _ in range(quarter_turns % 4):
            c = [c[3], c[0], c[1], c[2]]
        return Quad.from_corners(c)

    @property
    def area(self) -> float:
        c = self.corners
        return abs(sum(c[i].x * c[(i + 1) % 4].y - c[(i + 1) % 4].x * c[i].y for i in range(4))) / 2

    @property
    def is_convex(self) -> bool:
        c = self.corners
        sign = 0.0
        for i in range(4):
            a, b, d = c[i], c[(i + 1) % 4], c[(i + 2) % 4]
            cross = (b.x - a.x) * (d.y - b.y) - (b.y - a.y) * (d.x - b.x)
            if abs(cross) < 1e-12:
                return False
            if sign == 0:
                sign = cross
            elif (cross > 0) != (sign > 0):
                return False
        return True

    @property
    def interior_angles(self) -> list[float]:
        c = self.corners
        angles = []
        for i in range(4):
            prev, cur, nxt = c[(i + 3) % 4], c[i], c[(i + 1) % 4]
            ux, uy = prev.x - cur.x, prev.y - cur.y
            vx, vy = nxt.x - cur.x, nxt.y - cur.y
            lengths = math.hypot(ux, uy) * math.hypot(vx, vy)
            if lengths == 0:
                angles.append(0.0)
                continue
            cosine = max(-1.0, min(1.0, (ux * vx + uy * vy) / lengths))
            angles.append(math.degrees(math.acos(cosine)))
        return angles

    @property
    def bounding_box(self) -> Rect:
        xs = [p.x for p in self.corners]
        ys = [p.y for p in self.corners]
        return Rect(x=min(xs), y=min(ys), width=max(xs) - min(xs), height=max(ys) - min(ys))


class Homography(BaseModel):
    """Projective transform between planes, e.g. camera image → flat pack card."""

    model_config = ConfigDict(frozen=True)

    matrix: tuple[float, ...]  # row-major 3×3, normalized so the last element is 1 when possible

    @classmethod
    def from_matrix(cls, m: np.ndarray) -> Homography | None:
        m = np.asarray(m, dtype=float).reshape(9)
        if not np.all(np.isfinite(m)):
            return None
        scale = m[8] if abs(m[8]) > 1e-12 else 1.0
        return cls(matrix=tuple(float(v) for v in m / scale))

    @classmethod
    def mapping(cls, source: list[Point], destination: list[Point]) -> Homography | None:
        """Solves the transform mapping four source points onto four destination points."""
        if len(source) != 4 or len(destination) != 4:
            return None
        rows, rhs = [], []
        for s, d in zip(source, destination, strict=True):
            rows.append([s.x, s.y, 1, 0, 0, 0, -s.x * d.x, -s.y * d.x])
            rhs.append(d.x)
            rows.append([0, 0, 0, s.x, s.y, 1, -s.x * d.y, -s.y * d.y])
            rhs.append(d.y)
        a = np.array(rows, dtype=float)
        if abs(np.linalg.det(a)) < 1e-12:
            return None
        h = np.linalg.solve(a, np.array(rhs, dtype=float))
        return cls.from_matrix(np.append(h, 1.0))

    @property
    def array(self) -> np.ndarray:
        return np.array(self.matrix, dtype=float).reshape(3, 3)

    def apply(self, p: Point) -> Point | None:
        m = self.matrix
        w = m[6] * p.x + m[7] * p.y + m[8]
        if abs(w) < 1e-12:
            return None
        return Point(x=(m[0] * p.x + m[1] * p.y + m[2]) / w, y=(m[3] * p.x + m[4] * p.y + m[5]) / w)

    def apply_many(self, points: np.ndarray) -> np.ndarray:
        """Maps an (n, 2) array; rows on the line at infinity become NaN."""
        pts = np.hstack([points, np.ones((len(points), 1))]) @ self.array.T
        w = pts[:, 2:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = pts[:, :2] / w
        out[np.abs(w[:, 0]) < 1e-12] = np.nan
        return out

    @property
    def inverse(self) -> Homography | None:
        a = self.array
        if abs(np.linalg.det(a)) < 1e-12:
            return None
        return Homography.from_matrix(np.linalg.inv(a))
