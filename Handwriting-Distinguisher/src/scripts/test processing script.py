import cv2
import os
import numpy as np

# Configuration Directories
INPUT_DIR = "C:\\Users\\BC-Tech\\Documents\\Chibueze's Code\\Personal-Projects\\Handwriting-Distinguisher\\data\\raw\\raw_images"
OUTPUT_DIR = "C:\\Users\\BC-Tech\\Documents\\Chibueze's Code\\Personal-Projects\\Handwriting-Distinguisher\\data\\processed"
OUTPUT_C = "C:\\Users\\BC-Tech\\Documents\\Chibueze's Code\\Personal-Projects\\Handwriting-Distinguisher\\data\\processed\\HAMS_C"
OUTPUT_K = "C:\\Users\\BC-Tech\\Documents\\Chibueze's Code\\Personal-Projects\\Handwriting-Distinguisher\\data\\processed\\HAMS_K"
OUTPUT_Z = "C:\\Users\\BC-Tech\\Documents\\Chibueze's Code\\Personal-Projects\\Handwriting-Distinguisher\\data\\processed\\HAMS_Z"

TARGET_SIZE = (128, 128)
MARGIN = 0.15 # 15% margin around detected handwriting
EDGE_MARGIN = 0.12 # 12% margin from edges to discard components

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img = cv2.imread(os.path.join(INPUT_DIR, filename))
    
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur + Otsu's thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # connected components (find separate handwriting blobs)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    h, w = gray.shape
    keep_mask = np.zeros_like(thresh)

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]

        cx, cy = centroids[i]

        # discard small noise
        if area < 50:
            continue

        # discard components near edges (likely labels / marks)
        if (
            cx < EDGE_MARGIN * w
            or cx > (1 - EDGE_MARGIN) * w
            or cy < EDGE_MARGIN * h
            or cy > (1 - EDGE_MARGIN) * h
        ):
            continue

        keep_mask[labels == i] = 255

    coords = cv2.findNonZero(keep_mask)

    if coords is None:
        cropped = gray
    else:
        x, y, w2, h2 = cv2.boundingRect(coords)

        mx = int(w2 * MARGIN)
        my = int(h2 * MARGIN)

        x = max(0, x - mx)
        y = max(0, y - my)
        w2 = min(gray.shape[1] - x, w2 + 2 * mx)
        h2 = min(gray.shape[0] - y, h2 + 2 * my)

        cropped = gray[y:y+h2, x:x+w2]

    # resize to square with padding
    ch, cw = cropped.shape
    scale = min(TARGET_SIZE[0] / cw, TARGET_SIZE[1] / ch)
    nw, nh = int(cw * scale), int(ch * scale)

    resized = cv2.resize(cropped, (nw, nh))
    canvas = np.ones(TARGET_SIZE, dtype="uint8") * 255

    xo = (TARGET_SIZE[0] - nw) // 2
    yo = (TARGET_SIZE[1] - nh) // 2
    canvas[yo:yo+nh, xo:xo+nw] = resized

    # Smart saving based on filename
    if "HAMS-C" in filename:
        cv2.imwrite(os.path.join(OUTPUT_C, filename), canvas)
    elif "HAMS-K" in filename:
        cv2.imwrite(os.path.join(OUTPUT_K, filename), canvas)
    elif "HAMS-Z" in filename:
        cv2.imwrite(os.path.join(OUTPUT_Z, filename), canvas)
    else:
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), canvas)

print("Done.")
