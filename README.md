# PostRec version 1.0.0
### _powered by Festus_

---
<img src="Assets/posture-rec.webp" width="100%">

Desktop posture-monitoring prototype that uses only OpenCV for real-time posture estimation.
No datasets or external ML models — purely geometric/contour heuristics.

You also get a dataset from the process which can be used later for more improvemenys if needed

## test 1 image
<img src="Assets/test1 (1).png">

## Features
- PyQt6 GUI with live camera feed
- OpenCV contour-based posture heuristics (spine angle, shoulder tilt, head offset)
- Real-time overlay and status indicator (Optimal / Adjust / Critical)
- Session recording and CSV export

## test 2 image
<img src="Assets/test1 (2).png">

## Requirements
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py
```
## test 3 image
<img src="Assets/test1 (3).png">
## Notes
- Works best with plain backgrounds and decent lighting.
- This is a heuristic-based prototype for demonstration and should not be used as a medical device.
