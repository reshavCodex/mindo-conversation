"""Rolling average over last N frames to reduce flicker (Phase 11)."""
from collections import deque
import numpy as np


class TemporalSmoother:
    def __init__(self, classes, window_size=8):
        self.classes = classes
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, prob_vector):
        """prob_vector: np.array of shape (num_classes,)"""
        self.buffer.append(prob_vector)
        avg = np.mean(self.buffer, axis=0)
        smoothed_label = self.classes[int(np.argmax(avg))]
        return smoothed_label, avg

    def reset(self):
        self.buffer.clear()
