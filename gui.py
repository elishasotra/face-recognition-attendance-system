import tkinter as tk
from PIL import Image, ImageTk
import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

# ---------------- DATASET ----------------
path = "dataset"
images = []
classNames = []

for file in os.listdir(path):
    img = cv2.imread(f"{path}/{file}")
    if img is not None:
        images.append(img)
        classNames.append(os.path.splitext(file)[0].upper())


# ---------------- ENCODING ----------------
def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)

        if len(encodes) > 0:
            encodeList.append(encodes[0])

    return encodeList


encodeListKnown = findEncodings(images)

# ---------------- ATTENDANCE LOCK ----------------
already_marked = []


# ---------------- ATTENDANCE ----------------
def markAttendance(name):

    if name not in already_marked:

        with open("attendance.csv", "a+") as f:

            now = datetime.now()
            dt = now.strftime("%H:%M:%S")

            f.writelines(f"\n{name},{dt}")

        already_marked.append(name)


# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)


def start_camera():
    update_frame()


def update_frame():

    success, frame = cap.read()

    if success:

        small = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_locations(small)
        encodes = face_recognition.face_encodings(small, faces)

        for encodeFace, faceLoc in zip(encodes, faces):

            matches = face_recognition.compare_faces(
                encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(
                encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            # -------- KNOWN FACE --------
            if matches[matchIndex] and faceDis[matchIndex] < 0.5:

                name = classNames[matchIndex]

                markAttendance(name)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.rectangle(frame, (x1, y2-50), (x2, y2),
                              (0, 255, 0), cv2.FILLED)

                cv2.putText(frame, name,
                            (x1+6, y2-25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2)

                cv2.putText(frame, "MARKED",
                            (x1+6, y2-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),
                            2)

            # -------- UNKNOWN FACE --------
            else:

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                cv2.rectangle(frame, (x1, y2-40), (x2, y2),
                              (0, 0, 255), cv2.FILLED)

                cv2.putText(frame, "UNKNOWN",
                            (x1+6, y2-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    window.after(10, update_frame)


# ---------------- ADD NEW FACE ----------------
def add_new_face():

    name = name_entry.get().upper()

    if name == "":
        print("Enter name first")
        return

    cam = cv2.VideoCapture(0)

    while True:

        ret, frame = cam.read()

        cv2.putText(frame, "Press S to Save Face",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1)

        if key == ord("s"):

            cv2.imwrite(f"dataset/{name}.jpg", frame)

            print(f"{name} added to dataset")

            break

        if key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


# ---------------- GUI ----------------
window = tk.Tk()

window.title("Face Recognition Attendance System")

window.geometry("900x650")

window.configure(bg="#1e1e1e")

title = tk.Label(window,
                 text="Face Recognition Attendance System",
                 font=("Arial", 22, "bold"),
                 bg="#1e1e1e",
                 fg="white")

title.pack(pady=15)

video_label = tk.Label(window)
video_label.pack(pady=10)

# -------- Name Entry --------
name_entry = tk.Entry(window,
                      font=("Arial", 14),
                      width=20)

name_entry.pack(pady=10)
name_entry.insert(0, "Enter Name")

# -------- Buttons Frame --------
btn_frame = tk.Frame(window, bg="#1e1e1e")
btn_frame.pack(pady=20)

start_btn = tk.Button(btn_frame,
                      text="Start Camera",
                      font=("Arial", 14, "bold"),
                      bg="#27ae60",
                      fg="white",
                      width=15,
                      command=start_camera)

start_btn.grid(row=0, column=0, padx=10)

add_btn = tk.Button(btn_frame,
                    text="Add New Face",
                    font=("Arial", 14, "bold"),
                    bg="#2980b9",
                    fg="white",
                    width=15,
                    command=add_new_face)

add_btn.grid(row=0, column=1, padx=10)

exit_btn = tk.Button(btn_frame,
                     text="Exit",
                     font=("Arial", 14, "bold"),
                     bg="#c0392b",
                     fg="white",
                     width=15,
                     command=window.destroy)

exit_btn.grid(row=0, column=2, padx=10)

window.mainloop()
