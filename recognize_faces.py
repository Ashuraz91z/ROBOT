import cv2
import numpy as np

# Charger le modèle entraîné
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_model.yml")

# Dictionnaire mis à jour avec des noms réels
label_dict = {
    0: "Lucas",   # Remplace "Person1" par le nom réel
    1: "Rachel"   # Remplace "Person2" par un autre nom
}

# Classificateur Haar pour détecter les visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# URL du flux vidéo ESP32-CAM
url = "http://192.168.4.1/Test"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Erreur : Impossible de se connecter au flux vidéo.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire les frames.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label_id, confidence = face_recognizer.predict(face)

        if confidence < 100:  # Seuil de confiance
            label = label_dict.get(label_id, "Inconnu")
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            label = "Inconnu"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Reconnaissance Faciale", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()