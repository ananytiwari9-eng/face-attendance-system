import cv2
import os
import time
import sys


# ==========================================
# GET STUDENT DETAILS
# ==========================================

if len(sys.argv) >= 3:

    student_id = sys.argv[1].strip()
    student_name = sys.argv[2].strip()

else:

    student_id = input(
        "Enter Student ID: "
    ).strip()

    student_name = input(
        "Enter Student Name: "
    ).strip()


# ==========================================
# VALIDATION
# ==========================================

if not student_id or not student_name:

    print("Student ID aur Name required hai!")
    exit()


# ==========================================
# STUDENT FOLDER
# ==========================================

folder_name = (
    f"{student_id}_{student_name}"
)

save_path = os.path.join(
    "dataset",
    folder_name
)

os.makedirs(
    save_path,
    exist_ok=True
)


# ==========================================
# EXISTING IMAGES
# ==========================================

existing_images = []

for filename in os.listdir(save_path):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        try:

            number = int(
                os.path.splitext(filename)[0]
            )

            existing_images.append(number)

        except ValueError:

            pass


if existing_images:

    count = max(existing_images)

else:

    count = 0


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not camera.isOpened():

    print(
        "Camera open nahi ho raha!"
    )

    exit()


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


# ==========================================
# FACE DETECTOR
# ==========================================

face_detector = cv2.CascadeClassifier(

    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"

)


# ==========================================
# SETTINGS
# ==========================================

TARGET_IMAGES = count + 30

CAPTURE_DELAY = 1.0

last_capture_time = 0


print("\n================================")
print("   FACE REGISTRATION")
print("================================")

print(
    "Student ID:",
    student_id
)

print(
    "Student Name:",
    student_name
)

print(
    "Existing images:",
    count
)

print(
    "30 new images capture hongi."
)

print(
    "Q dabakar stop kar sakte ho."
)

print("================================")


# ==========================================
# MAIN LOOP
# ==========================================

while True:

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
    # FACE DETECTION
    # --------------------------------------

    faces = face_detector.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(100, 100)

    )


    # ======================================
    # MULTIPLE FACES
    # ======================================

    if len(faces) > 1:

        cv2.putText(

            frame,

            "Only ONE face allowed",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 0, 255),

            2

        )


    # ======================================
    # ONE FACE
    # ======================================

    elif len(faces) == 1:

        x, y, w, h = faces[0]


        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            (255, 0, 0),

            2

        )


        current_time = time.time()


        # ----------------------------------
        # CAPTURE
        # ----------------------------------

        if (
            current_time - last_capture_time
            >= CAPTURE_DELAY
        ):

            count += 1


            face = gray[
                y:y + h,
                x:x + w
            ]


            filename = os.path.join(

                save_path,

                f"{count}.jpg"

            )


            success = cv2.imwrite(

                filename,

                face

            )


            if success:

                last_capture_time = (
                    current_time
                )

                print(
                    f"Image {count} saved"
                )


        cv2.putText(

            frame,

            "Face detected",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2

        )


    # ======================================
    # NO FACE
    # ======================================

    else:

        cv2.putText(

            frame,

            "Face camera ke saamne lao",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 0, 255),

            2

        )


    # ======================================
    # COUNTER
    # ======================================

    new_images = count - (
        max(existing_images)
        if existing_images
        else 0
    )

    cv2.putText(

        frame,

        f"New Images: {new_images}/30",

        (20, 80),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 255, 255),

        2

    )


    cv2.putText(

        frame,

        student_name,

        (20, 120),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2

    )


    # ======================================
    # SHOW
    # ======================================

    cv2.imshow(

        "Student Face Registration",

        frame

    )


    # ======================================
    # KEY
    # ======================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        print(
            "\nRegistration stopped."
        )

        break


    # ======================================
    # 30 NEW IMAGES
    # ======================================

    if new_images >= 30:

        print(
            "\n30 new images captured! ✅"
        )

        break


# ==========================================
# CLOSE
# ==========================================

camera.release()

cv2.destroyAllWindows()


print("\n================================")
print(" REGISTRATION COMPLETE")
print("================================")

print(
    "Student ID:",
    student_id
)

print(
    "Student Name:",
    student_name
)

print(
    "Total images:",
    count
)

print(
    "Folder:",
    save_path
)

print("================================")