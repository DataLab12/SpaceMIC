import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import math
import seaborn as sns
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from pathlib import Path

# Directory containing images
image_folder = '../Corrosion_full_data/images'
output_folder = './Sift_Convex_Hull'

# Ensure output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load all images from the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.tif', '.jpeg'))]
images = [cv2.imread(os.path.join(image_folder, img)) for img in image_files if cv2.imread(os.path.join(image_folder, img)) is not None]
gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect and compute keypoints and descriptors
all_descriptors = []
for gray in gray_images:


    kp, desc = sift.detectAndCompute(gray, None)
    if desc is not None:
        all_descriptors.append(desc)

# Concatenate all descriptors from all images if there are any
if all_descriptors:
    all_descriptors = np.vstack(all_descriptors).astype(np.float32)
    # Clustering descriptors across all images
    kmeans = KMeans(n_clusters=11, random_state=123).fit(all_descriptors)
else:
    print("No descriptors found in any images. Check the images or adjust SIFT parameters.")
    exit()

# Define minimum contour area to keep
min_contour_area = 4500  # Adjust this value based on specific requirements

# Assign descriptors in each image to clusters and create segmentation masks
for i, (img, gray) in enumerate(zip(images, gray_images)):

    kp, desc = sift.detectAndCompute(gray, None)
    if desc is not None:
        desc = desc.astype(np.float32)
        labels = kmeans.predict(desc)
        points = np.array([k.pt for k in kp])

        # Prepare data for DBSCAN
        data = np.hstack((points, labels[:, np.newaxis]))
        dbscan = DBSCAN(eps=15, min_samples=11)
        cluster_labels = dbscan.fit_predict(data)

        # Visualize the clustering results
        output_image = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        mask_image = np.zeros_like(gray, dtype=np.uint8)  # Create a mask for drawing contours

        cluster_points = {}
        for label, point in zip(cluster_labels, points):
            if label != -1:
                if label in cluster_points:
                    cluster_points[label].append(point)
                else:
                    cluster_points[label] = [point]

        # Draw convex hulls and prepare mask
        for label, pts in cluster_points.items():
            pts = np.array(pts, dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.drawContours(mask_image, [hull], -1, 255, -1)  # Draw filled convex hull on mask

        # Find and filter contours based on area
        contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) >= min_contour_area:
                cv2.drawContours(output_image, [contour], -1, (255, 255, 255), -1)  # Draw filtered contours

        output_path = os.path.join(output_folder, os.path.basename(image_files[i]))
        cv2.imwrite(output_path, output_image)

    

# Evaluation Results
GT_FOLDER = Path("../Corrosion_full_data/masks")
PRED_FOLDER = Path('./Sift_Convex_Hull')

FLIGHT_INDICATOR = 'flight_indicator'

THRESHOLD = 127
# RESIZE_TO = (512, 512)

def open_img(path):
    img = Image.open(str(path)).convert("L")
    return img

def threshold(img):
    np_array = np.array(img)
    return (np_array >= THRESHOLD).astype(np.uint8) * 255

data = []
for file in GT_FOLDER.glob("*"):
    filename = os.path.split(file)[-1].replace(".png", ".png")

    # Process gt image
    gt = open_img(file)
    gt = threshold(gt)
    gt_binary = gt == 255
    gt_mask_area = np.sum(gt_binary)

    # Process pred image
    pred = open_img(PRED_FOLDER / filename)
    pred = threshold(pred)
    pred_binary = pred == 255
    pred_mask_area = np.sum(pred_binary)

    # Confusion matrix elements
    TP = np.sum(np.logical_and(gt_binary, pred_binary))
    TN = np.sum(np.logical_and(~gt_binary, ~pred_binary))
    FP = np.sum(np.logical_and(~gt_binary, pred_binary))
    FN = np.sum(np.logical_and(gt_binary, ~pred_binary))


    intersect = TP
    union = np.sum(np.logical_or(gt_binary, pred_binary))
    iou = intersect / union if union != 0 else 0
    dice = 2 * intersect / (gt_mask_area + pred_mask_area) if (gt_mask_area + pred_mask_area) != 0 else 0

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    specificity = TN / (TN + FP) if (TN + FP) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Flatten masks for AUC calculation
    try:
        auc = roc_auc_score(gt_binary.flatten(), pred_binary.flatten())
    except ValueError:
        auc = np.nan  # When only one class present in y_true, AUC is not defined

    # Print individual metrics
    print("Sample:", filename)
    print("IoU:", iou)
    print("Dice Score:", dice)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("Specificity:", specificity)
    print("F1 Score:", f1)
    print("AUC:", auc)
    print()

    data.append({
        FLIGHT_INDICATOR: filename,
        'iou': iou,
        'dice': dice,
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'specificity': specificity,
        'f1': f1,
        'auc': auc,
    })

# Compute mean metrics
df = pd.DataFrame(data)
print("\nMean Result:")
for metric in ['iou', 'dice', 'accuracy', 'recall', 'precision', 'specificity', 'f1', 'auc']:
    mean_val = df[metric].mean(skipna=True)
    print(f"Mean {metric}:", mean_val)
