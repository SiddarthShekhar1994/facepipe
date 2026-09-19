"""Drawing pipeline results onto a frame, for the window and the dashboard alike."""

import cv2

from facepipe.types import FaceResult, Frame

KNOWN = (0, 200, 0)
UNKNOWN = (0, 0, 220)


def draw_results(frame: Frame, results: list[FaceResult]) -> None:
    """A box per face with the best match and its similarity, or "unknown", drawn in place."""
    for r in results:
        x1, y1, x2, y2 = r.detection.bbox.round().astype(int).tolist()
        if r.matches:
            color, label = KNOWN, f"{r.matches[0].name} {r.matches[0].similarity:.2f}"
        else:
            color, label = UNKNOWN, "unknown"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
