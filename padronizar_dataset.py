import cv2
import mediapipe as mp
import numpy as np
import os

IMG_SIZE = 100  # padroniza todas para 100x100
INPUT_DIR = "dataset_raw"        # onde suas fotos originais estão
OUTPUT_DIR = "dataset_padronizado"  # onde salvar padronizadas

os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def process_image(path, output_path):
    img = cv2.imread(path)
    h, w, _ = img.shape

    # Detecta rostos
    results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not results.detections:
        print("Nenhum rosto encontrado em:", path)
        return

    detection = results.detections[0]
    box = detection.location_data.relative_bounding_box

    # Bounding box relativo → absoluto
    x = int(box.xmin * w)
    y = int(box.ymin * h)
    bw = int(box.width * w)
    bh = int(box.height * h)

    # Ajusta o ROI para evitar cortes ruins
    x = max(0, x - bw//6)
    y = max(0, y - bh//6)
    bw = min(w - x, bw + bw//3)
    bh = min(h - y, bh + bh//3)

    face = img[y:y+bh, x:x+bw]

    # Converte para grayscale
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Redimensiona para dimensões fixas
    face_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))

    # Normaliza (0–255)
    face_norm = cv2.equalizeHist(face_resized)

    # Salva
    cv2.imwrite(output_path, face_norm)
    print("Salvo:", output_path)

# LOOP PRINCIPAL

for person in os.listdir(INPUT_DIR):
    person_folder = os.path.join(INPUT_DIR, person)
    if not os.path.isdir(person_folder):
        continue

    # pasta de saída
    out_folder = os.path.join(OUTPUT_DIR, person)
    os.makedirs(out_folder, exist_ok=True)

    for filename in os.listdir(person_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            in_path = os.path.join(person_folder, filename)
            out_path = os.path.join(out_folder, filename)

            process_image(in_path, out_path)
