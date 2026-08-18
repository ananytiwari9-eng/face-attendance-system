import cv2
import os
import json
import numpy as np


# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = "dataset"
MODEL_PATH = "trainer.yml"
NAMES_PATH = "names.json"


# ==========================================
# FACE DETECTOR
# ==========================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ==========================================
# READ DATASET FOLDERS
# ==========================================

faces = []
labels = []

names = {}

label_number = 1


print("\n================================")
print(" FACE ATTENDANCE TRAINING")
print("================================\n")


if not os.path.exists(DATASET_PATH):

    print("Dataset folder nahi mila!")

    exit()


folders = sorted(
    [
        folder
        for folder in os.listdir(DATASET_PATH)
        if os.path.isdir(
            os.path.join(DATASET_PATH, folder)
        )
    ]
)


if not folders:

    print("Dataset empty hai!")

    exit()


# ==========================================
# PROCESS EACH STUDENT
# ==========================================

for folder_name in folders:

    folder_path = os.path.join(
        DATASET_PATH,
        folder_name
    )


    # Expected:
    # 052004_Ananya Tiwari

    parts = folder_name.split("_", 1)


    if len(parts) != 2:

        print(
            "Invalid folder name:",
            folder_name
        )

        continue


    student_id = parts[0]
    student_name = parts[1]


    # Save mapping

    names[str(label_number)] = {
        "id": student_id,
        "name": student_name
    }


    print(
        f"Training: {student_name} "
        f"(ID: {student_id})"
    )


    image_files = os.listdir(folder_path)


    student_image_count = 0


    for image_file in image_files:

        image_path = os.path.join(
            folder_path,
            image_file
        )


        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )


        if image is None:

            continue


        detected_faces = face_detector.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )


        for (x, y, w, h) in detected_faces:

            face = image[y:y+h, x:x+w]


            faces.append(face)

            labels.append(label_number)


            student_image_count += 1


    print(
        f"  Faces found: {student_image_count}"
    )


    label_number += 1


# ==========================================
# CHECK TRAINING DATA
# ==========================================

if len(faces) == 0:

    print("\nKoi face training data nahi mila!")

    exit()


print("\n--------------------------------")
print("Training model...")
print("--------------------------------")


# ==========================================
# TRAIN LBPH
# ==========================================

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(
    faces,
    np.array(labels)
)


# ==========================================
# SAVE MODEL
# ==========================================

recognizer.write(
    MODEL_PATH
)


# ==========================================
# SAVE NAMES
# ==========================================

with open(
    NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        names,
        file,
        indent=4,
        ensure_ascii=False
    )


# ==========================================
# RESULT
# ==========================================

print("\n================================")
print(" TRAINING COMPLETE ✅")
print("================================")

print(
    "\nModel saved:",
    MODEL_PATH
)

print(
    "Names saved:",
    NAMES_PATH
)

print(
    "Total faces trained:",
    len(faces)
)

print("\nStudents:")

for label, student in names.items():

    print(
        f"{label} → "
        f"{student['id']} → "
        f"{student['name']}"
    )

print("\nAb attendance_system.py run kar sakte ho.")