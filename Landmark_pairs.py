import cv2
import dlib
import pandas as pd
import numpy as np
from scipy.spatial import procrustes
from tqdm import tqdm
import os

# Load Dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) > 0:
        landmarks = predictor(gray, faces[0])
        return [(p.x, p.y) for p in landmarks.parts()]
    return None

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks, dtype=np.float64)
    mean = np.mean(landmarks, axis=0)
    landmarks -= mean
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    landmarks /= max_dist
    return landmarks.flatten()

def procrustes_distance(l1, l2):
    l1 = l1.reshape(-1, 2)
    l2 = l2.reshape(-1, 2)
    _, _, disparity = procrustes(l1, l2)
    return disparity

def estimate_pose(image, landmarks):
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye
        (225.0, 170.0, -135.0),      # Right eye
        (-150.0, -150.0, -125.0),    # Left mouth
        (150.0, -150.0, -125.0)      # Right mouth
    ])
    image_points = np.array([
        landmarks[30], landmarks[8], landmarks[36],
        landmarks[45], landmarks[48], landmarks[54]
    ], dtype="double")

    size = (image.shape[1], image.shape[0])
    focal = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    return np.array(rvec), np.array(tvec)

def compute_similarity(lm1, lm2, pose1, pose2):
    landmark_dist = procrustes_distance(lm1, lm2)
    pose_dist = np.linalg.norm(pose1[0] - pose2[0]) + np.linalg.norm(pose1[1] - pose2[1])
    return landmark_dist + pose_dist

# Load dataset
dataset = pd.read_csv('./stressid_data/source_1fps.txt', sep=' ', names=['path', 'label'])
dataset['subject'] = dataset['path'].apply(lambda p: os.path.basename(os.path.dirname(os.path.dirname(p))))

# Extract landmarks and poses
landmarks_dict = {}
pose_dict = {}

for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Extracting Features"):
    img = cv2.imread(row['path'])
    if img is None:
        print(f"Could not load image: {row['path']}")
        continue
    landmarks = get_landmarks(img)
    if landmarks:
        try:
            norm_landmarks = normalize_landmarks(landmarks)
            pose = estimate_pose(img, landmarks)
            landmarks_dict[row['path']] = norm_landmarks
            pose_dict[row['path']] = pose
        except:
            print(f"Pose estimation failed for: {row['path']}")

# Pair images from different labels and subjects
with open('./stressid_data/source_LM_pairs.txt', 'w') as f:
    for i, row1 in tqdm(dataset.iterrows(), total=len(dataset), desc="Finding Pairs"):
        path1, label1, subj1 = row1['path'], row1['label'], row1['subject']
        if path1 not in landmarks_dict:
            continue

        min_dist = float('inf')
        best_path = None

        for j, row2 in dataset.iterrows():
            if i == j:
                continue
            path2, label2, subj2 = row2['path'], row2['label'], row2['subject']

            if label1 == label2:
                continue  # Skip same label
            if subj1 == subj2:
                continue  # Skip same subject
            if path2 not in landmarks_dict:
                continue

            dist = compute_similarity(
                landmarks_dict[path1], landmarks_dict[path2],
                pose_dict[path1], pose_dict[path2]
            )
            if dist < min_dist:
                min_dist = dist
                best_path = path2

        if best_path:
            f.write(f"{path1} {best_path} {min_dist:.6f}\n")
            f.flush()
