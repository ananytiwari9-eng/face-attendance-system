import cv2
import json
import mysql.connector
from datetime import datetime
from collections import deque, Counter


# ==========================================
# MYSQL SETTINGS
# ==========================================

DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "Prince@1234"
DB_NAME = "face_attendance"


# ==========================================
# RECOGNITION SETTINGS
# ==========================================

# Lower LBPH confidence = better match
CONFIDENCE_THRESHOLD = 70

# Same prediction kitne frames tak confirm karna hai
CONFIRMATION_FRAMES = 5

# 5 frames me minimum kitne same hone chahiye
MIN_CONFIRMATIONS = 4


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
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
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

def mark_attendance(student_id, student_name):

    today = datetime.now().date()

    # Check whether attendance already exists today
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

    # Already marked
    if result:

        print(
            f"Already marked today: "
            f"{student_name} ({student_id})"
        )

        return False

    # Current time
    current_time = datetime.now().time()

    # Insert attendance
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


# Camera resolution
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


last_confirmed_label = None

attendance_marked_labels = set()


# ==========================================
# START SYSTEM
# ==========================================

print("\n================================")
print(" FACE ATTENDANCE SYSTEM")
print("================================")

print("Camera started...")
print("Q dabakar exit karo.")

print(
    f"Confirmation: "
    f"{MIN_CONFIRMATIONS}/{CONFIRMATION_FRAMES} frames"
)

print(
    f"Confidence threshold: "
    f"{CONFIDENCE_THRESHOLD}"
)


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    # --------------------------------------
    # READ CAMERA
    # --------------------------------------

    ret, frame = camera.read()

    if not ret or frame is None:

        print(
            "Camera se frame nahi mil raha!"
        )

        continue


    # --------------------------------------
    # GRAYSCALE
    # --------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------
    # DETECT FACES
    # --------------------------------------

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )


    # --------------------------------------
    # NO FACE
    # --------------------------------------

    if len(faces) == 0:

        prediction_history.clear()

        last_confirmed_label = None

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

        face = gray[
            y:y+h,
            x:x+w
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


        student = names.get(
            str(label)
        )


        # ----------------------------------
        # ACCEPT ONLY GOOD MATCH
        # ----------------------------------

        if (
            student is not None
            and confidence < CONFIDENCE_THRESHOLD
        ):

            current_label = str(label)

            prediction_history.append(
                current_label
            )

            print(
                "Prediction:",
                student["name"],
                "| Confidence:",
                round(confidence, 2)
            )


        else:

            prediction_history.append(
                "unknown"
            )

            print(
                "Prediction: Unknown",
                "| Confidence:",
                round(confidence, 2)
            )


        # ==================================
        # CHECK 5-FRAME CONFIRMATION
        # ==================================

        confirmed_label = None


        if len(prediction_history) >= CONFIRMATION_FRAMES:

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


                if count >= MIN_CONFIRMATIONS:

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

                student_id = confirmed_student["id"]

                student_name = confirmed_student["name"]


                last_confirmed_label = (
                    confirmed_label
                )


                # ------------------------------
                # MARK ATTENDANCE ONLY ONCE
                # ------------------------------

                if confirmed_label not in attendance_marked_labels:

                    mark_attendance(
                        student_id,
                        student_name
                    )

                    attendance_marked_labels.add(
                        confirmed_label
                    )


                # ------------------------------
                # GREEN BOX
                # ------------------------------

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )


                # ------------------------------
                # NAME
                # ------------------------------

                cv2.putText(
                    frame,
                    student_name,
                    (x, y-35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


                # ------------------------------
                # ID
                # ------------------------------

                cv2.putText(
                    frame,
                    f"ID: {student_id}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


                # ------------------------------
                # CONFIRMED
                # ------------------------------

                cv2.putText(
                    frame,
                    "CONFIRMED",
                    (x, y+h+25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


        # ==================================
        # NOT CONFIRMED
        # ==================================

        else:

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 0, 255),
                2
            )


            cv2.putText(
                frame,
                "Checking...",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


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