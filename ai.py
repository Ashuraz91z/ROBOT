import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

# Dossier pour sauvegarder les visages connus
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Charger le modèle LBPH (si existe)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
model_file = "face_model.yml"
if os.path.exists(model_file):
    face_recognizer.read(model_file)
    print("Modèle chargé avec succès.")
else:
    print("Aucun modèle trouvé. Un nouveau sera créé.")

# Classificateur Haar pour détecter les visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# URL du flux vidéo
VIDEO_URL = "http://192.168.4.1/Test"

# Interface graphique principale
def start_gui():
    root = tk.Tk()
    root.title("Capture et Entraînement des Visages")
    
    # Créer un label pour afficher le flux vidéo
    video_label = tk.Label(root)
    video_label.pack()

    # Fonction pour afficher le flux vidéo
    def show_video():
        cap = cv2.VideoCapture(VIDEO_URL)
        if not cap.isOpened():
            messagebox.showerror("Erreur", "Impossible d'accéder au flux vidéo.")
            return
        
        def update_frame():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                video_label.imgtk = imgtk
                video_label.configure(image=imgtk)
                video_label.after(10, update_frame)
            else:
                cap.release()
        
        update_frame()

    # Lancer la capture de visages
    def capture_and_train():
        name = simpledialog.askstring("Nom de la personne", "Entrez le nom de la personne :")
        if not name:
            messagebox.showerror("Erreur", "Nom invalide.")
            return

        save_path = os.path.join(KNOWN_FACES_DIR, name)
        os.makedirs(save_path, exist_ok=True)
        cap = cv2.VideoCapture(VIDEO_URL)
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Erreur", "Impossible de lire le flux vidéo.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                count += 1
                face_file = os.path.join(save_path, f"face_{count}.png")
                cv2.imwrite(face_file, face)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Capture {count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Capture des Visages", frame)
            if count >= 10:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        train_model()
        messagebox.showinfo("Succès", f"Visages de {name} capturés et modèle mis à jour.")

    def train_model():
        faces = []
        labels = []
        label_id = 0

        for person_name in os.listdir(KNOWN_FACES_DIR):
            person_path = os.path.join(KNOWN_FACES_DIR, person_name)
            if os.path.isdir(person_path):
                for file in os.listdir(person_path):
                    if file.endswith(".png") or file.endswith(".jpg"):
                        img_path = os.path.join(person_path, file)
                        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        faces.append(gray_img)
                        labels.append(label_id)
                label_id += 1

        if faces and labels:
            face_recognizer.train(faces, np.array(labels))
            face_recognizer.write(model_file)
            print("Modèle sauvegardé avec succès.")
        else:
            print("Aucune donnée pour l'entraînement.")

    # Reconnaissance faciale
    def recognize_faces():
        cap = cv2.VideoCapture(VIDEO_URL)
        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Erreur", "Impossible de lire le flux vidéo.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                label_id, confidence = face_recognizer.predict(face)
                label = "Inconnu"
                if confidence < 100:
                    label = f"ID {label_id}, Conf {confidence:.2f}"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Reconnaissance Faciale", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # Boutons de l'interface
    tk.Button(root, text="Afficher le Flux Vidéo", command=show_video, padx=10, pady=5).pack(pady=10)
    tk.Button(root, text="Capturer et Entraîner Visages", command=capture_and_train, padx=10, pady=5).pack(pady=10)
    tk.Button(root, text="Reconnaître Visages", command=recognize_faces, padx=10, pady=5).pack(pady=10)
    tk.Button(root, text="Quitter", command=root.quit, padx=10, pady=5).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    start_gui()