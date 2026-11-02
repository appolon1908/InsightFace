"""Model-independent source tests. No model is loaded or certified."""
import importlib
import os
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


class SyntheticDetector:
    def __init__(self):
        self.faces = np.array([[1, 2, 3, 4, 0], [5, 6, 8, 9, 0]], dtype=np.float32)
        self.input_size = None

    def setInputSize(self, size):
        self.input_size = size

    def detect(self, image):
        return None, self.faces


class SyntheticRecognizer:
    def alignCrop(self, image, face):
        return image

    def feature(self, image):
        return np.array([[3.0, 4.0]], dtype=np.float32)


detector = SyntheticDetector()
recognizer = SyntheticRecognizer()
constructors = []
def detector_constructor(*args):
    constructors.append(('detector', args[0]))
    return detector
def recognizer_constructor(*args):
    constructors.append(('recognizer', args[0]))
    return recognizer

os.environ['DETECTOR_MODEL'] = '/synthetic-fixture/no-real-detector-model'
os.environ['RECOGNIZER_MODEL'] = '/synthetic-fixture/no-real-recognizer-model'
with patch.object(cv2, 'FaceDetectorYN', SimpleNamespace(create=detector_constructor)), \
     patch.object(cv2, 'FaceRecognizerSF', SimpleNamespace(create=recognizer_constructor)):
    main = importlib.import_module('app.main')


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def reset_synthetic_detector():
    detector.faces = np.array([[1, 2, 3, 4, 0], [5, 6, 8, 9, 0]], dtype=np.float32)


@pytest.fixture
def image():
    import base64
    ok, encoded = cv2.imencode('.png', np.zeros((12, 16, 3), dtype=np.uint8))
    assert ok
    return base64.b64encode(encoded).decode()
