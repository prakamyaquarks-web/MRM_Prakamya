# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


#!/usr/bin/env python3
"""
Real-time ArUco marker detection + pose estimation (position, orientation, depth)
using OpenCV and a webcam.

Supports 4x4, 5x5 and 6x6 ArUco dictionaries (as generated at https://chev.me/arucogen).

Usage examples
--------------
    python aruco_pose_estimation.py --size 0.05              # 5 cm marker, all families
    python aruco_pose_estimation.py --size 0.05 --dict 4X4_50
    python aruco_pose_estimation.py --size 0.05 --calib calib.npz --camera 1

Keys:  q / ESC = quit,   s = save screenshot,   d = cycle dictionary

Install:  pip install opencv-contrib-python numpy
"""

import argparse
import time

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Dictionaries
# --------------------------------------------------------------------------- #
# The ArUco dictionaries of one family are nested (ID 5 in 4X4_50 is the same
# pattern as ID 5 in 4X4_1000), so the "_1000" dictionary of each family covers
# any tag you generate on chev.me (50 / 100 / 250 / 1000).
ARUCO_DICTS = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "4X4_250": cv2.aruco.DICT_4X4_250,
    "4X4_1000": cv2.aruco.DICT_4X4_1000,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "5X5_250": cv2.aruco.DICT_5X5_250,
    "5X5_1000": cv2.aruco.DICT_5X5_1000,
    "6X6_50": cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100,
    "6X6_250": cv2.aruco.DICT_6X6_250,
    "6X6_1000": cv2.aruco.DICT_6X6_1000,
}
# Default: scan all three families at once
DEFAULT_FAMILIES = ["4X4_1000", "5X5_1000", "6X6_1000"]

NEW_API = hasattr(cv2.aruco, "ArucoDetector")  # OpenCV >= 4.7


def make_detector(name):
    """Return a callable detect(gray) -> (corners, ids) for a dictionary name."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[name])
    if NEW_API:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        det = cv2.aruco.ArucoDetector(dictionary, params)

        def detect(gray):
            corners, ids, _ = det.detectMarkers(gray)
            return corners, ids
    else:
        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        def detect(gray):
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
            return corners, ids

    return detect


# --------------------------------------------------------------------------- #
# Camera intrinsics
# --------------------------------------------------------------------------- #
def load_intrinsics(calib_path, width, height):
    """Load camera matrix / distortion from .npz, or fall back to an approximation."""
    if calib_path:
        data = np.load(calib_path)
        cam = data["camera_matrix"] if "camera_matrix" in data else data["mtx"]
        dist = data["dist_coeffs"] if "dist_coeffs" in data else data["dist"]
        print(f"[INFO] Loaded calibration from {calib_path}")
        return cam.astype(np.float64), dist.astype(np.float64)

    # Rough pinhole approximation (~60-70 deg horizontal FOV).
    # Depth values will be approximate; calibrate your camera for accurate results.
    f = float(width)
    cam = np.array([[f, 0, width / 2.0],
                    [0, f, height / 2.0],
                    [0, 0, 1.0]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    print("[WARN] No calibration file given -> using approximate intrinsics. "
          "Distances/angles will be approximate.")
    return cam, dist


# --------------------------------------------------------------------------- #
# Pose estimation
# --------------------------------------------------------------------------- #
def estimate_pose(corners, marker_size, cam_matrix, dist_coeffs):
    """Pose of one marker via solvePnP (works on all OpenCV versions)."""
    h = marker_size / 2.0
    # Corner order returned by ArUco: TL, TR, BR, BL (marker centre = origin)
    obj_points = np.array([[-h, h, 0],
                           [h, h, 0],
                           [h, -h, 0],
                           [-h, -h, 0]], dtype=np.float32)
    img_points = corners.reshape(4, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, cam_matrix, dist_coeffs,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return ok, rvec, tvec


def rotation_to_euler(rvec):
    """Rodrigues vector -> (roll, pitch, yaw) in degrees (X, Y, Z axes)."""
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def put_text(img, text, org, color=(0, 255, 0), scale=0.5):
    """Text with a dark outline for readability."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_marker_info(frame, corners, marker_id, family, rvec, tvec, euler,
                     cam_matrix, dist_coeffs, marker_size):
    pts = corners.reshape(4, 2).astype(int)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    cv2.circle(frame, tuple(pts[0]), 5, (0, 0, 255), -1)  # top-left corner marker

    # 3D axes (X red, Y green, Z blue)
    cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, marker_size * 0.6, 2)

    x, y, z = tvec.flatten()
    dist = float(np.linalg.norm(tvec))
    roll, pitch, yaw = euler

    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    lines = [
        f"ID {marker_id} [{family}]",
        f"Depth Z: {z * 100:.1f} cm  Dist: {dist * 100:.1f} cm",
        f"Pos X:{x * 100:.1f} Y:{y * 100:.1f} cm",
        f"Rot R:{roll:.0f} P:{pitch:.0f} Y:{yaw:.0f} deg",
    ]
    for i, line in enumerate(lines):
        put_text(frame, line, (x0, max(15, y0 - 10 - 16 * (len(lines) - 1 - i))))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="ArUco pose estimation (4x4, 5x5, 6x6)")
    ap.add_argument("--size", type=float, default=0.05,
                    help="Physical side length of the marker's black square, in METRES "
                         "(default 0.05 = 5 cm). Measure your printed tag!")
    ap.add_argument("--dict", default="ALL", choices=["ALL"] + list(ARUCO_DICTS),
                    help="Dictionary to use (default ALL = 4x4 + 5x5 + 6x6)")
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    ap.add_argument("--calib", default=None,
                    help=".npz with 'camera_matrix' and 'dist_coeffs' from camera calibration")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_matrix, dist_coeffs = load_intrinsics(args.calib, w, h)

    # Dictionary modes: ALL, then each individual dictionary (cycle with 'd')
    modes = ["ALL"] + list(ARUCO_DICTS)
    mode_idx = modes.index(args.dict)
    detectors = {}

    def get_detectors(mode):
        names = DEFAULT_FAMILIES if mode == "ALL" else [mode]
        for n in names:
            if n not in detectors:
                detectors[n] = make_detector(n)
        return [(n, detectors[n]) for n in names]

    prev_t, fps = time.time(), 0.0
    print("[INFO] Running. Press 'q' or ESC to quit, 's' to save a screenshot, 'd' to change dictionary.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Failed to read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mode = modes[mode_idx]
        n_found = 0

        for family, detect in get_detectors(mode):
            corners, ids = detect(gray)
            if ids is None:
                continue
            for c, mid in zip(corners, ids.flatten()):
                ok_pose, rvec, tvec = estimate_pose(c, args.size, cam_matrix, dist_coeffs)
                if not ok_pose:
                    continue
                euler = rotation_to_euler(rvec)
                draw_marker_info(frame, c, int(mid), family, rvec, tvec, euler,
                                 cam_matrix, dist_coeffs, args.size)
                n_found += 1

        # HUD
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
        prev_t = now
        put_text(frame, f"Dict: {mode}   Marker size: {args.size * 100:.1f} cm   "
                        f"Detected: {n_found}   FPS: {fps:.1f}", (10, 25), (255, 255, 0), 0.6)

        cv2.imshow("ArUco Pose Estimation", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            fname = f"aruco_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Saved {fname}")
        elif key == ord("d"):
            mode_idx = (mode_idx + 1) % len(modes)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()