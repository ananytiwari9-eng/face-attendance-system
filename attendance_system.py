import os
import json
import cv2
import mysql.connector

from dotenv import load_dotenv
from datetime import datetime
from collections import deque, Counter


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv()


# ==========================================
# MYSQL SETTINGS
# ==========================================

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")


# ==========================================
# RECOGNITION SETTINGS
# ==========================================

# LBPH:
# Lower confidence/distance = better match.
#
# 50 is stricter than the previous 70.
CONFIDENCE_THRESHOLD = 50

# Kitne consecutive frames ka history rakhna hai
CONFIRMATION_FRAMES = 7

# 7 frames me minimum 6 same predictions chahiye
MIN_CONFIRMATIONS = 6


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

try:

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read("trainer.yml")

    print("Face model loaded successfully! ✅")

except Exception as error:

    print("Face model load nahi ho raha!")
    print(error)

    exit()


# ==========================================
# LOAD NAMES
# ==========================================

try:

    with open(
        "names.json",
        "r",
        encoding="utf-8"
    ) as file:

        names = json.load(file)

    print("Student names loaded successfully! ✅")

except Exception as error:

    print("names.json load nahi ho raha!")
    print(error)

    exit()


# ==========================================
# FACE DETECTOR
# ==========================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


if face_detector.empty():

    print("Face detector load nahi hua!")

    exit()


# ==========================================
# MYSQL CONNECTION
# ==========================================

try:

    connection = mysql.connector.connect(
        host=DB_HOST,
        port=3306,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

    cursor = connection.cursor()

    print("MySQL connected successfully! ✅")

except mysql.connector.Error as error:

    print("MySQL connection failed!")
    print(error)

    exit()


# ==========================================
# ATTENDANCE FUNCTION
# ==========================================

def mark_attendance(
    student_id,
    student_name
):

    today = datetime.now().date()


    # --------------------------------------
    # CHECK DUPLICATE
    # --------------------------------------

    check_query = """
        SELECT id
        FROM attendance
        WHERE student_id = %s
        AND attendance_date = %s
    """


    cursor.execute(
        check_query,
        (
            student_id,
            today
        )
    )


    result = cursor.fetchone()


    if result:

        print(
            f"Already marked today: "
            f"{student_name} ({student_id})"
        )

        return False


    # --------------------------------------
    # CURRENT TIME
    # --------------------------------------

    current_time = datetime.now().time()


    # --------------------------------------
    # INSERT ATTENDANCE
    # --------------------------------------

    insert_query = """
        INSERT INTO attendance
        (
            student_id,
            name,
            attendance_date,
            attendance_time,
            status
        )
        VALUES
        (
            %s,
            %s,
            %s,
            %s,
            %s
        )
    """


    values = (
        student_id,
        student_name,
        today,
        current_time,
        "Present"
    )


    cursor.execute(
        insert_query,
        values
    )


    connection.commit()


    print("\n================================")
    print(" Attendance marked! ✅")
    print("================================")

    print("ID:", student_id)
    print("Name:", student_name)
    print("Date:", today)
    print("Time:", current_time)

    print("================================\n")


    return True


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


if not camera.isOpened():

    print("Camera open nahi ho raha!")

    cursor.close()
    connection.close()

    exit()


# ==========================================
# CAMERA RESOLUTION
# ==========================================

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


print("Camera successfully connected! ✅")


# ==========================================
# FACE CONFIRMATION STORAGE
# ==========================================

prediction_history = deque(
    maxlen=CONFIRMATION_FRAMES
)


attendance_marked_labels = set()


# ==========================================
# START SYSTEM
# ==========================================

print("\n================================")
print(" FACE ATTENDANCE SYSTEM")
print("================================")

print("Camera started...")
print("Q dabakar band karo.")

print(
    f"Confirmation: "
    f"{MIN_CONFIRMATIONS}/"
    f"{CONFIRMATION_FRAMES} frames"
)

print(
    f"Confidence threshold: "
    f"{CONFIDENCE_THRESHOLD}"
)


# ==========================================
# MAIN LOOP
# ==========================================

while True:


    # ======================================
    # READ CAMERA
    # ======================================

    ret, frame = camera.read()


    if not ret or frame is None:

        print(
            "Camera se frame nahi mil raha!"
        )

        continue


    # ======================================
    # GRAYSCALE
    # ======================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ======================================
    # FACE DETECTION
    # ======================================

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(100, 100)
    )


    # ======================================
    # NO FACE
    # ======================================

    if len(faces) == 0:

        prediction_history.clear()

        cv2.putText(
            frame,
            "No face detected",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # ======================================
    # PROCESS FACES
    # ======================================

    for (x, y, w, h) in faces:


        # ----------------------------------
        # FACE CROP
        # ----------------------------------

        face = gray[
            y:y + h,
            x:x + w
        ]


        # ----------------------------------
        # RECOGNITION
        # ----------------------------------

        try:

            label, confidence = recognizer.predict(
                face
            )

        except Exception as error:

            print(
                "Recognition error:",
                error
            )

            continue


        # ----------------------------------
        # FIND STUDENT
        # ----------------------------------

        student = names.get(
            str(label)
        )


        # ==================================
        # STRICT MATCH
        # ==================================

        if (
            student is not None
            and confidence <= CONFIDENCE_THRESHOLD
        ):

            current_label = str(label)


            prediction_history.append(
                current_label
            )


            print(
                "Possible match:",
                student["name"],
                "| Distance:",
                round(confidence, 2)
            )


        else:

            prediction_history.append(
                "unknown"
            )


            print(
                "Unknown face",
                "| Distance:",
                round(confidence, 2)
            )


        # ==================================
        # CHECK CONFIRMATION
        # ==================================

        confirmed_label = None


        if (
            len(prediction_history)
            >= CONFIRMATION_FRAMES
        ):


            valid_predictions = [

                item

                for item in prediction_history

                if item != "unknown"

            ]


            if valid_predictions:


                counter = Counter(
                    valid_predictions
                )


                most_common_label, count = (
                    counter.most_common(1)[0]
                )


                # ----------------------------------
                # STRICT CONFIRMATION
                # ----------------------------------

                if count >= MIN_CONFIRMATIONS:

                    # Make sure unknown frames
                    # are not dominating the history

                    unknown_count = prediction_history.count(
                        "unknown"
                    )


                    if unknown_count <= 1:

                        confirmed_label = (
                            most_common_label
                        )


        # ==================================
        # CONFIRMED STUDENT
        # ==================================

        if confirmed_label is not None:


            confirmed_student = names.get(
                confirmed_label
            )


            if confirmed_student is not None:


                student_id = (
                    confirmed_student["id"]
                )


                student_name = (
                    confirmed_student["name"]
                )


                # ----------------------------------
                # MARK ATTENDANCE ONLY ONCE
                # ----------------------------------

                if (
                    confirmed_label
                    not in attendance_marked_labels
                ):

                    mark_attendance(
                        student_id,
                        student_name
                    )


                    attendance_marked_labels.add(
                        confirmed_label
                    )


                # ----------------------------------
                # GREEN BOX
                # ----------------------------------

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )


                # ----------------------------------
                # NAME
                # ----------------------------------

                cv2.putText(
                    frame,
                    student_name,
                    (x, y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


                # ----------------------------------
                # ID
                # ----------------------------------

                cv2.putText(
                    frame,
                    f"ID: {student_id}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


                # ----------------------------------
                # CONFIRMED
                # ----------------------------------

                cv2.putText(
                    frame,
                    "CONFIRMED",
                    (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


        # ==================================
        # UNKNOWN / NOT CONFIRMED
        # ==================================

        else:


            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )


            cv2.putText(
                frame,
                "UNKNOWN",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


        # ----------------------------------
        # PROCESS FIRST FACE ONLY
        # ----------------------------------

        break


    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        "Face Attendance System",
        frame
    )


    # ======================================
    # Q = EXIT
    # ======================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ==========================================
# CLOSE EVERYTHING
# ==========================================

camera.release()

cv2.destroyAllWindows()


try:

    cursor.close()

    connection.close()

except:

    pass


print(
    "\nAttendance system closed."
)