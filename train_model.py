import cv2
import os
import numpy as np

# Chemin du dossier contenant les images des visages connus
known_faces_dir = "known_faces"
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Classificateur Haar pour détecter les visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

labels = []
faces = []
label_id = 0
label_dict = {}

for root, dirs, files in os.walk(known_faces_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            # Détecter le visage dans l'image
            face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in face:
                faces.append(img[y:y+h, x:x+w])
                labels.append(label_id)
            if label not in label_dict:
                label_dict[label_id] = label
                label_id += 1

# Entraîner le modèle avec les visages et leurs labels
face_recognizer.train(faces, np.array(labels))

# Sauvegarder le modèle
face_recognizer.write("face_model.yml")
print("Modèle entraîné et sauvegardé.")