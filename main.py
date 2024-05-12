import logging
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow

from deepface import DeepFace
from deepface.modules import (
    demography,
    detection,
    modeling,
    preprocessing,
    recognition,
    representation,
    streaming,
    verification,
)
from ui.main_window import Ui_MainWindow

FPS: int = int(1000 / 25)
DATABASE_PATH: str = "database"


@dataclass
class IdentityResult:
    identified_image_path: str
    identified_image: np.ndarray
    distance: float
    threshold: float


class DeepFaceHelper:
    def __init__(
        self,
        db_path,
        model_name="VGG-Face",
        detector_backend="opencv",
        distance_metric="cosine",
        source=0,
        frame_threshold=30,
    ):
        self.db_path = db_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        self.source = source
        self.frame_threshold = frame_threshold
        self.cap = cv2.VideoCapture(self.source + cv2.CAP_DSHOW)  # webcam (default 0)
        self.num_frames_with_faces = 0

    def face_analysis(self) -> None:
        streaming.build_facial_recognition_model(model_name=self.model_name)

        # call a dummy find function for db_path once to create embeddings before start capturing
        _ = streaming.search_identity(
            detected_face=np.zeros([224, 224, 3]),  # random numbers
            db_path=self.db_path,
            detector_backend=self.detector_backend,
            distance_metric=self.distance_metric,
            model_name=self.model_name,
        )

        face_found = False

        window.face_analysis_timer.timeout.connect(
            lambda: self.capture_camera_and_analyze_face(face_found)
        )
        window.face_analysis_timer.start(FPS)

    def capture_camera_and_analyze_face(self, face_found: bool) -> None:
        """
        Capture camera and analyze face
        """
        has_frame, img = self.cap.read()
        if not has_frame:
            logging.error("Could not read frame from camera")
            return

        raw_img = img.copy()

        if not face_found:
            faces_coordinates = streaming.grab_facial_areas(
                img, detector_backend=self.detector_backend
            )

            detected_faces = streaming.extract_facial_areas(
                img=img, faces_coordinates=faces_coordinates
            )
            img = streaming.highlight_facial_areas(
                img=img, faces_coordinates=faces_coordinates
            )

            self.num_frames_with_faces = (
                self.num_frames_with_faces + 1 if len(faces_coordinates) else 0
            )

            face_found = (
                self.num_frames_with_faces > 0
                and self.num_frames_with_faces % self.frame_threshold == 0
            )
            if face_found:
                # add analyze results into img - derive from raw_img
                img = streaming.highlight_facial_areas(
                    img=raw_img, faces_coordinates=faces_coordinates
                )

                img = self.perform_facial_recognition_details(
                    img=img,
                    faces_coordinates=faces_coordinates,
                    detected_faces=detected_faces,
                    db_path=self.db_path,
                    detector_backend=self.detector_backend,
                    distance_metric=self.distance_metric,
                    model_name=self.model_name,
                )

                window.face_analysis_timer.stop()
                QTimer.singleShot(
                    200, lambda: window.show_opencv_img(img)
                )  # Ensure that right image is shown (QTimer act asynchronously)

        window.show_opencv_img(img)

    @staticmethod
    def search_identity_details(
        detected_face: np.ndarray,
        db_path: str,
        model_name: str,
        detector_backend: str,
        distance_metric: str,
    ) -> IdentityResult | None:
        """
        Search an identity in facial database.
        Args:
            detected_face (np.ndarray): extracted individual facial image
            db_path (string): Path to the folder containing image files. All detected faces
                in the database will be considered in the decision-making process.
            model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
                OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
            detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
                'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
                (default is opencv).
            distance_metric (string): Metric for measuring similarity. Options: 'cosine',
                'euclidean', 'euclidean_l2' (default is cosine).
        Returns:
            identity_result (IdentityResult): identified image path, identified image, distance and threshold
        """
        target_path = None
        try:
            dfs = DeepFace.find(
                img_path=detected_face,
                db_path=db_path,
                model_name=model_name,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
                enforce_detection=False,
                silent=True,
            )
        except ValueError as err:
            if f"No item found in {db_path}" in str(err):
                logging.warning(
                    f"No item is found in {db_path}."
                    "So, no facial recognition analysis will be performed."
                )
                dfs = []
            else:
                raise err
        if len(dfs) == 0:
            return None

        # detected face is coming from parent, safe to access 1st index
        df = dfs[0]
        if df.shape[0] == 0:
            return None

        candidate = df.iloc[0]
        identity_result = IdentityResult(
            identified_image_path=candidate["identity"],
            identified_image=None,
            distance=candidate["distance"],
            threshold=candidate["threshold"],
        )

        # load found identity image - extracted if possible
        target_objs = DeepFace.extract_faces(
            img_path=identity_result.identified_image_path,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True,
        )

        # extract facial area of the identified image if and only if it has one face
        # otherwise, show image as is
        if len(target_objs) == 1:
            # extract 1st item directly
            target_obj = target_objs[0]
            target_img = target_obj["face"]
            target_img = cv2.resize(
                target_img,
                (streaming.IDENTIFIED_IMG_SIZE, streaming.IDENTIFIED_IMG_SIZE),
            )
            target_img *= 255
            target_img = target_img[:, :, ::-1]
        else:
            target_img = cv2.imread(identity_result.identified_image_path)

        identity_result.identified_image = target_img

        return identity_result

    def perform_facial_recognition_details(
        self,
        img,
        faces_coordinates,
        detected_faces,
        db_path=None,
        detector_backend=None,
        distance_metric=None,
        model_name=None,
    ):
        """
        Perform facial recognition
        Args:
            img (np.ndarray): image itself
            detected_faces (list): list of extracted detected face images as numpy
            faces_coordinates (list): list of facial area coordinates as tuple with
                x, y, w and h values
            db_path (string): Path to the folder containing image files. All detected faces
                in the database will be considered in the decision-making process.
            detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
                'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
                (default is opencv).
            distance_metric (string): Metric for measuring similarity. Options: 'cosine',
                'euclidean', 'euclidean_l2' (default is cosine).
            model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
                OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
        Returns:
            img (np.ndarray): image with identified face informations
        """
        db_path = db_path or self.db_path
        detector_backend = detector_backend or self.detector_backend
        distance_metric = distance_metric or self.distance_metric
        model_name = model_name or self.model_name

        for idx, (x, y, w, h) in enumerate(faces_coordinates):
            detected_face = detected_faces[idx]
            identity_result = DeepFaceHelper.search_identity_details(
                detected_face=detected_face,
                db_path=db_path,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
                model_name=model_name,
            )
            if identity_result is None:
                continue

            img = streaming.overlay_identified_face(
                img=img,
                target_img=identity_result.identified_image,
                label=identity_result.identified_image_path,
                x=x,
                y=y,
                w=w,
                h=h,
            )

        return img


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.verify_pushbutton.clicked.connect(self.verify_pushbutton_click)
        self.ui.camera_label.setPixmap(
            QPixmap("./ui/resources/deepface-icon-labeled.png")
        )

        self.face_analysis_timer = QTimer(self)
        self.deep_face_helper = DeepFaceHelper(DATABASE_PATH)

    def verify_pushbutton_click(self):
        self.deep_face_helper.face_analysis()

    def show_opencv_img(self, img):
        convert = QImage(img, img.shape[1], img.shape[0], QImage.Format.Format_BGR888)
        self.ui.camera_label.setPixmap(QPixmap.fromImage(convert))


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
