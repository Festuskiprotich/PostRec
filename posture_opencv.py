"""Heuristic posture estimation using OpenCV contour processing."""
import cv2
import numpy as np
import math

def analyze_contour(contour, img_shape):
    """Given the largest contour assume it's the person silhouette.
    Returns metrics: spine_angle (deg from vertical), shoulder_tilt_px, head_offset_px, bbox"""
    if contour is None or len(contour) == 0:
        return None

    x,y,w,h = cv2.boundingRect(contour)
    bbox = (x,y,w,h)
    # Create mask for contour
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # --- Head offset: topmost contour point within bbox ---
    ys = contour[:,0,1]
    min_y = int(np.min(ys))
    top_points = contour[contour[:,0,1] == min_y][:,0]
    # take median x
    head_x = int(np.median(top_points[:,0]))
    center_x = x + w//2
    head_offset = abs(head_x - center_x)

    # --- Shoulder detection: search for left/right highest points in upper region of bbox ---
    upper_limit = y + int(h * 0.35)
    left_region = mask[y:upper_limit, x:x + w//2]
    right_region = mask[y:upper_limit, x + w//2:x + w]
    left_pts = cv2.findNonZero(left_region)
    right_pts = cv2.findNonZero(right_region)
    left_shoulder_y = None
    right_shoulder_y = None
    left_shoulder_x = None
    right_shoulder_x = None
    if left_pts is not None:
        # coordinates relative to region -> convert to image coords
        pts = left_pts.reshape(-1,2)
        top_idx = np.argmin(pts[:,1])
        ly = pts[top_idx,1] + y
        lx = pts[top_idx,0] + x
        left_shoulder_y = ly
        left_shoulder_x = lx
    if right_pts is not None:
        pts = right_pts.reshape(-1,2)
        top_idx = np.argmin(pts[:,1])
        ry = pts[top_idx,1] + y
        rx = pts[top_idx,0] + x + w//2
        right_shoulder_y = ry
        right_shoulder_x = rx

    shoulder_tilt = None
    if left_shoulder_y is not None and right_shoulder_y is not None:
        shoulder_tilt = abs(left_shoulder_y - right_shoulder_y)

    # --- Spine angle: fit line to contour points and compute angle to vertical ---
    pts = contour.reshape(-1,2).astype(np.float32)
    # Require enough points
    if pts.shape[0] >= 5:
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        angle_rad = math.atan2(vy, vx)
        angle_deg = math.degrees(angle_rad)
        # angle_deg is angle to horizontal; convert to angle to vertical
        angle_to_vertical = 90.0 - abs(angle_deg)
    else:
        angle_to_vertical = 0.0

    return {
        'spine_angle_deg': angle_to_vertical,   # 0 = perfectly vertical; bigger numbers -> tilted
        'shoulder_tilt_px': shoulder_tilt,
        'head_offset_px': head_offset,
        'bbox': bbox,
        'left_shoulder': (left_shoulder_x, left_shoulder_y) if left_shoulder_x else None,
        'right_shoulder': (right_shoulder_x, right_shoulder_y) if right_shoulder_x else None,
    }

def classify(metrics, img_height):
    """Simple thresholds to decide status."""
    if metrics is None:
        return 'Unknown'
    spine = metrics.get('spine_angle_deg', 0.0)
    shoulder = metrics.get('shoulder_tilt_px', None)
    head_off = metrics.get('head_offset_px', 0.0)

    # Normalize shoulder by image height (as percent)
    shoulder_pct = None
    if shoulder is not None:
        shoulder_pct = (shoulder / float(img_height)) * 100.0

    # thresholds (tunable)
    if spine > 12.0 or (shoulder_pct and shoulder_pct > 6.0) or head_off > 80:
        return 'Critical'
    if spine > 6.0 or (shoulder_pct and shoulder_pct > 3.0) or head_off > 40:
        return 'Adjust'
    return 'Optimal'
