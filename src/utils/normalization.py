import numpy as np

def normalize_landmarks_center(coords):
    coords = coords.reshape(-1, 3)
    center = np.mean(coords, axis=0)
    coords -= center
    norm = np.linalg.norm(coords)
    return coords.flatten() / norm if norm != 0 else coords.flatten()
