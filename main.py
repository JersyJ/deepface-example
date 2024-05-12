import logging
import sys
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow

from deepface import DeepFace
from deepface.modules import streaming
from ui.main_window import Ui_MainWindow

FPS: int = 20
INTERVAL: int = int(1000 / FPS)
DATABASE_PATH: str = "database"

REGISTER_FPS: int = 30
REGISTER_INTERVAL: int = int(1000 / FPS)


class VerifyButton(Enum):
    NOT_VERIFYING = 0
    VERIFYING = 1


@dataclass
class IdentityResult:
    identified_image_path: str
    identified_image: np.ndarray | None
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
        _ = DeepFaceHelper.search_identity_details(

            detected_face=np.zeros([224, 224, 3]),  # random numbers
            db_path=self.db_path,
            detector_backend=self.detector_backend,
            distance_metric=self.distance_metric,
            model_name=self.model_name,
        )

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

                identity = self.perform_facial_recognition_details(
                    img=img,
                    faces_coordinates=faces_coordinates,
                    detected_faces=detected_faces,
                    db_path=self.db_path,
                    detector_backend=self.detector_backend,
                    distance_metric=self.distance_metric,
                    model_name=self.model_name,
                )

                if not identity:
                    logging.info("No identity found.")
                    face_found = False
                    self.num_frames_with_faces = 0
                else:
                    img = identity.identified_image
                    window.stop_verify_identified()
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
    ) -> IdentityResult | None:
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
            identity_result (Identity Result) | None: identified image path, identified image, distance and threshold
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
            if not identity_result:
                continue

            identity_result.identified_image = streaming.overlay_identified_face(
                img=img,
                target_img=identity_result.identified_image,
                label=identity_result.identified_image_path,
                x=x,
                y=y,
                w=w,
                h=h,
            )

        return identity_result

    def capture_camera(self):
        has_frame, img = self.cap.read()
        if not has_frame:
            logging.error("Could not read frame from camera")

        window.show_opencv_img(img)

    def register_face(self):
        pass

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.verify_pushbutton.clicked.connect(self.verify_pushbutton_click)
        self.ui.register_pushbutton.clicked.connect(self.register_pushbutton_click)
        self.reset_camera_label()
        self.face_analysis_timer = QTimer(self)
        self.face_analysis_timer.timeout.connect(
            lambda: self.deep_face_helper.capture_camera_and_analyze_face(False)
        )
        self.register_timer = QTimer(self)
        self.register_timer.timeout.connect(
            lambda: self.deep_face_helper.capture_camera
        )
        self.deep_face_helper = DeepFaceHelper(DATABASE_PATH)
        self.verify_state = VerifyButton.NOT_VERIFYING

    def show_opencv_img(self, img):
        convert = QImage(img, img.shape[1], img.shape[0], QImage.Format.Format_BGR888)
        self.ui.camera_label.setPixmap(QPixmap.fromImage(convert))

    def reset_verify_button(self):
        self.ui.verify_pushbutton.setText("Verify")
        self.ui.verify_pushbutton.setStyleSheet('QPushButton {color: white;}')

    def reset_camera_label(self):
        self.ui.camera_label.setPixmap(
            QPixmap("./ui/resources/deepface-icon-labeled.png")
        )

    def start_verify_manually(self):
        self.deep_face_helper.face_analysis()

        self.face_analysis_timer.start(INTERVAL)

        self.verify_state = VerifyButton.VERIFYING
        self.ui.verify_pushbutton.setText("Cancel Verification")
        self.ui.verify_pushbutton.setStyleSheet('QPushButton {color: yellow;}')

    def stop_verify_identified(self):
        self.face_analysis_timer.stop()
        self.verify_state = VerifyButton.NOT_VERIFYING
        self.reset_verify_button()

    def stop_verify_manually(self):
        self.face_analysis_timer.stop()
        self.verify_state = VerifyButton.NOT_VERIFYING
        self.reset_verify_button()
        self.reset_camera_label()

    def verify_pushbutton_click(self):
        if self.verify_state == VerifyButton.NOT_VERIFYING:
            self.start_verify_manually()
        elif self.verify_state == VerifyButton.VERIFYING:
            self.stop_verify_manually()

    def register_pushbutton_click(self):
        self.register_timer.start(REGISTER_INTERVAL)


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
