import tkinter as tk
import os

def start_system():
    os.system("python3 attendance_system.py")

window = tk.Tk()
window.title("Face Recognition Attendance System")
window.geometry("400x300")

title = tk.Label(window,
                 text="Face Recognition Attendance",
                 font=("Arial",16,"bold"))
title.pack(pady=30)

start_button = tk.Button(window,
                         text="Start Attendance",
                         font=("Arial",14),
                         bg="green",
                         fg="white",
                         command=start_system)

start_button.pack(pady=20)

exit_button = tk.Button(window,
                        text="Exit",
                        font=("Arial",14),
                        bg="red",
                        fg="white",
                        command=window.destroy)

exit_button.pack(pady=20)

window.mainloop()