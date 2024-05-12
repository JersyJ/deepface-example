import logging
import sys

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap

from deepface import DeepFace

from deepface.modules import (
    modeling,
    representation,
    verification,
    recognition,
    demography,
    detection,
    streaming,
    preprocessing,
)

from ui.main_window import Ui_MainWindow


class DeepFaceHelper():
    def __init__(self, db_path, model_name="VGG-Face", detector_backend="opencv", distance_metric="cosine", source=0,
                 frame_threshold=5):
        self.db_path = db_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        self.source = source
        self.frame_threshold = frame_threshold
        self.cap = cv2.VideoCapture(self.source)  # webcam (default 0)

    def face_analysis(self) -> None:
        streaming.build_facial_recognition_model(model_name=self.model_name)
        #
        # call a dummy find function for db_path once to create embeddings before starting webcam
        _ = streaming.search_identity(
            detected_face=np.zeros([224, 224, 3]),
            db_path=self.db_path,
            detector_backend=self.detector_backend,
            distance_metric=self.distance_metric,
            model_name=self.model_name,
        )

        face_found_img = None
        face_found = False

        window.face_analysis_timer.timeout.connect(
            lambda: self.capture_camera_and_analyze_face(face_found_img, face_found))
        window.face_analysis_timer.start(100)

    def capture_camera_and_analyze_face(self, face_found_img, face_found: bool) -> None:
        """
        Capture camera and analyze face
        """
        has_frame, img = self.cap.read()
        if not has_frame:
            logging.error("Could not read frame from camera")
            return

        window.show_opencv_img(img)

        # raw_img = img.copy()
        #
        # if not face_found:
        #     faces_coordinates = streaming.grab_facial_areas(img, detector_backend=detector_backend)
        #


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.verify_pushButton.clicked.connect(self.verify_pushButton_click)
        self.face_analysis_timer = QTimer(self)
        self.deep_face_helper = DeepFaceHelper("database")

    def verify_pushButton_click(self):
        self.deep_face_helper.face_analysis()

    def show_opencv_img(self, img):
        convert = QImage(img, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_BGR888)
        self.ui.camera_label.setPixmap(QPixmap.fromImage(convert))


def main() -> None:
    #DeepFace.stream("database", enable_face_analysis=False)
    pass


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
