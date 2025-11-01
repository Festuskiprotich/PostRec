"""PyQt6 + OpenCV posture sentinel (OpenCV-only detection)"""
import sys, time, csv, os
from PyQt6 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
from posture_opencv import analyze_contour, classify

class VideoThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object, object)  # QImage, metrics
    def __init__(self, cam_index=0, parent=None):
        super().__init__(parent)
        self.cam_index = cam_index
        self.running = False
        self.cap = None
    def run(self):
        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW if sys.platform.startswith('win') else 0)
        # Basic warm-up
        self.running = True
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            # preprocess to find largest contour (assume person)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7,7), 0)
            _,th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # invert if background bright
            if np.sum(th==255) > th.size*0.6:
                th = cv2.bitwise_not(th)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            person_contour = None
            if contours:
                # choose largest with sufficient area
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                for c in contours:
                    if cv2.contourArea(c) > 2000:  # ignore small
                        person_contour = c
                        break
            metrics = analyze_contour(person_contour, frame.shape) if person_contour is not None else None

            # Draw overlays on frame
            vis = frame.copy()
            if metrics and metrics.get('bbox'):
                x,y,w,h = metrics['bbox']
                # neon rectangle
                color = (40,255,180)
                cv2.rectangle(vis, (x,y), (x+w, y+h), color, 2, lineType=cv2.LINE_AA)
                # shoulders
                ls = metrics.get('left_shoulder')
                rs = metrics.get('right_shoulder')
                if ls:
                    cv2.circle(vis, (int(ls[0]), int(ls[1])), 6, (255,100,100), -1)
                if rs:
                    cv2.circle(vis, (int(rs[0]), int(rs[1])), 6, (255,100,100), -1)
                # head
                head_x = x + w//2
                head_y = y + int(0.12*h)
                cv2.circle(vis, (head_x, head_y), 6, (200,200,50), -1)
                # spine line
                mid_top = (x + w//2, y + int(0.28*h))
                mid_bottom = (x + w//2, y + h)
                cv2.line(vis, mid_top, mid_bottom, (120,200,255), 2, lineType=cv2.LINE_AA)
                # put metrics text
                cv2.putText(vis, f"Spine: {metrics.get('spine_angle_deg'):.1f} deg", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,255,200), 1, cv2.LINE_AA)
                if metrics.get('shoulder_tilt_px') is not None:
                    cv2.putText(vis, f"Shoulder tilt: {int(metrics.get('shoulder_tilt_px'))} px", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,255,200), 1, cv2.LINE_AA)

            # classification
            label = classify(metrics, frame.shape[0]) if metrics is not None else 'Unknown'
            # color mapping
            color = (0,255,0) if label=='Optimal' else (0,165,255) if label=='Adjust' else (0,0,255) if label=='Critical' else (200,200,200)
            cv2.putText(vis, f"Posture: {label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

            # convert BGR->RGB and to QImage
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
            self.frame_ready.emit(qimg, {'label': label, 'metrics': metrics})
            QtCore.QThread.msleep(30)
        if self.cap:
            self.cap.release()
    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Posture Sentinel — OpenCV Edition')
        self.resize(1100, 680)
        self.session_data = []
        self._setup_ui()
        self.video_thread = VideoThread(0)
        self.video_thread.frame_ready.connect(self.on_frame)
        self.video_thread.start()
        self.running = True

    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        # left metrics
        left = QtWidgets.QFrame()
        left.setFixedWidth(240)
        left.setObjectName('leftCard')
        left.setProperty('class', 'card')
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.addWidget(QtWidgets.QLabel('Real-time Metrics', objectName='title'))
        self.metric_spine = QtWidgets.QLabel('Spine: -')
        self.metric_shoulder = QtWidgets.QLabel('Shoulder tilt: -')
        self.metric_head = QtWidgets.QLabel('Head offset: -')
        left_layout.addWidget(self.metric_spine)
        left_layout.addWidget(self.metric_shoulder)
        left_layout.addWidget(self.metric_head)
        left_layout.addStretch()
        # center video
        center = QtWidgets.QFrame()
        center_layout = QtWidgets.QVBoxLayout(center)
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(640, 480)
        center_layout.addWidget(self.video_label)
        # controls
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton('Stop')
        self.start_btn.clicked.connect(self.toggle)
        self.export_btn = QtWidgets.QPushButton('Export CSV')
        self.export_btn.clicked.connect(self.export_csv)
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.export_btn)
        center_layout.addLayout(ctrl_layout)
        # right status
        right = QtWidgets.QFrame()
        right.setFixedWidth(260)
        right.setProperty('class', 'card')
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.addWidget(QtWidgets.QLabel('Posture Status', objectName='title'))
        self.status_label = QtWidgets.QLabel('Unknown', objectName='statusLabel')
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.status_label)
        self.session_overview = QtWidgets.QLabel('Session samples: 0\nOptimal: 0\nCritical: 0')
        right_layout.addWidget(self.session_overview)
        right_layout.addStretch()
        # assemble
        layout.addWidget(left)
        layout.addWidget(center, 1)
        layout.addWidget(right)
        # stylesheet
        try:
            with open('styles.qss','r') as f:
                self.setStyleSheet(f.read())
        except Exception:
            pass

    @QtCore.pyqtSlot(object, object)
    def on_frame(self, qimg, info):
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pix)
        label = info.get('label','Unknown')
        metrics = info.get('metrics') or {}
        self.status_label.setText(label)
        self.metric_spine.setText(f"Spine: {metrics.get('spine_angle_deg', 0.0):.1f} deg" if metrics else 'Spine: -')
        self.metric_shoulder.setText(f"Shoulder tilt: {int(metrics.get('shoulder_tilt_px'))} px" if metrics.get('shoulder_tilt_px') is not None else 'Shoulder tilt: -')
        self.metric_head.setText(f"Head offset: {int(metrics.get('head_offset_px'))} px" if metrics else 'Head offset: -')
        # append session sample
        self.session_data.append((time.time(), label, metrics))
        # update overview
        total = len(self.session_data)
        optimal = sum(1 for s in self.session_data if s[1]=='Optimal')
        critical = sum(1 for s in self.session_data if s[1]=='Critical')
        self.session_overview.setText(f"Session samples: {total}\nOptimal: {optimal}\nCritical: {critical}")

    def toggle(self):
        if self.running:
            self.video_thread.stop()
            self.running = False
            self.start_btn.setText('Start')
        else:
            self.video_thread = VideoThread(0)
            self.video_thread.frame_ready.connect(self.on_frame)
            self.video_thread.start()
            self.running = True
            self.start_btn.setText('Stop')

    def export_csv(self):
        out = 'posture_session.csv'
        with open(out, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp','label','spine_angle_deg','shoulder_tilt_px','head_offset_px'])
            for ts,label,metrics in self.session_data:
                writer.writerow([ts,label, metrics.get('spine_angle_deg') if metrics else '', metrics.get('shoulder_tilt_px') if metrics else '', metrics.get('head_offset_px') if metrics else ''])
        QtWidgets.QMessageBox.information(self, 'Export', f'CSV exported to {out}')

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
