import face_recognition
import os
import pickle

known_encodings = []
known_names = []

dataset_path = "dataset"

for file in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, file)
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)[0]

    known_encodings.append(encoding)
    known_names.append(os.path.splitext(file)[0])

with open("encodings.pkl", "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print("Encoding completed")