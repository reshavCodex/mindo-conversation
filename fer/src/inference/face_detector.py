"""Face detection + cropping using MediaPipe (Phase 9)."""
import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_detection_confidence=0.6):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence,
        )

    def detect_and_crop(self, frame_bgr):
        """Returns the largest detected face crop (BGR np.array) or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        if not results.detections:
            return None

        h, w, _ = frame_bgr.shape
        best_box, best_area = None, 0
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            x1 = max(int(box.xmin * w), 0)
            y1 = max(int(box.ymin * h), 0)
            bw = int(box.width * w)
            bh = int(box.height * h)
            area = bw * bh
            if area > best_area:
                best_area = area
                best_box = (x1, y1, bw, bh)

        if best_box is None:
            return None

        x1, y1, bw, bh = best_box
        x2, y2 = min(x1 + bw, w), min(y1 + bh, h)
        crop = frame_bgr[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def close(self):
        self.detector.close()
