# Posture Sentinel — OpenCV Only (PyQt6 UI)

Desktop posture-monitoring prototype that uses only OpenCV for real-time posture estimation.
No datasets or external ML models — purely geometric/contour heuristics.

## Features
- PyQt6 GUI with live camera feed
- OpenCV contour-based posture heuristics (spine angle, shoulder tilt, head offset)
- Real-time overlay and status indicator (Optimal / Adjust / Critical)
- Session recording and CSV export

## Requirements
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

## Notes
- Works best with plain backgrounds and decent lighting.
- This is a heuristic-based prototype for demonstration and should not be used as a medical device.
