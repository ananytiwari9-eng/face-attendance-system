from dotenv import load_dotenv
import os
load_dotenv()
from flask import Flask, render_template, redirect, url_for, request, Response
import mysql.connector
import subprocess
import sys
import os
import csv
import io

app = Flask(__name__)


# ==========================================
# MYSQL SETTINGS
# ==========================================

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )


# ==========================================
# DASHBOARD
# ==========================================

@app.route("/")
def dashboard():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT COUNT(*) AS total_students
            FROM students
        """)

        total_students = cursor.fetchone()["total_students"] or 0

        cursor.execute("""
            SELECT COUNT(DISTINCT student_id) AS present_today
            FROM attendance
            WHERE attendance_date = CURDATE()
            AND status = 'Present'
        """)

        present_today = cursor.fetchone()["present_today"] or 0

        absent_today = max(
            total_students - present_today,
            0
        )

        if total_students > 0:

            attendance_rate = round(
                (present_today / total_students) * 100,
                1
            )

        else:

            attendance_rate = 0

        cursor.execute("""
            SELECT
                student_id,
                name,
                attendance_date,
                attendance_time,
                status
            FROM attendance
            ORDER BY
                attendance_date DESC,
                attendance_time DESC
            LIMIT 10
        """)

        recent_attendance = cursor.fetchall()

        return render_template(
            "index.html",
            total_students=total_students,
            present_today=present_today,
            absent_today=absent_today,
            attendance_rate=attendance_rate,
            recent_attendance=recent_attendance
        )

    except mysql.connector.Error as error:

        print("Dashboard MySQL Error:", error)

        return f"""
        <h2>Database Error</h2>
        <p>{error}</p>
        """

    finally:

        if cursor:

            try:
                cursor.close()
            except:
                pass

        if connection:

            try:
                connection.close()
            except:
                pass


# ==========================================
# STUDENTS
# ==========================================

@app.route("/students", methods=["GET", "POST"])
def students():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        if request.method == "POST":

            student_id = request.form.get(
                "student_id",
                ""
            ).strip()

            student_name = request.form.get(
                "student_name",
                ""
            ).strip()

            if not student_id or not student_name:

                return redirect(
                    url_for("students")
                )

            cursor.execute("""
                SELECT id
                FROM students
                WHERE student_id = %s
            """, (student_id,))

            existing = cursor.fetchone()

            if existing:

                print(
                    "Student ID already exists:",
                    student_id
                )

                return redirect(
                    url_for("students")
                )

            cursor.execute("""
                INSERT INTO students
                (student_id, name)
                VALUES
                (%s, %s)
            """, (
                student_id,
                student_name
            ))

            connection.commit()

            print(
                "Student added successfully! ✅"
            )

            return redirect(
                url_for("students")
            )

        cursor.execute("""
            SELECT *
            FROM students
            ORDER BY id DESC
        """)

        student_list = cursor.fetchall()

        return render_template(
            "students.html",
            students=student_list
        )

    except mysql.connector.Error as error:

        print(
            "Students MySQL Error:",
            error
        )

        return f"""
        <h2>Database Error</h2>
        <p>{error}</p>
        """

    finally:

        if cursor:

            try:
                cursor.close()
            except:
                pass

        if connection:

            try:
                connection.close()
            except:
                pass


# ==========================================
# CAPTURE FACE
# ==========================================

@app.route("/capture-face/<student_id>")
def capture_face(student_id):

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute("""
            SELECT
                student_id,
                name
            FROM students
            WHERE student_id = %s
        """, (student_id,))

        student = cursor.fetchone()

        if not student:

            return redirect(
                url_for("students")
            )

        cursor.close()
        connection.close()

        cursor = None
        connection = None

        capture_script = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)
            ),
            "capture_faces.py"
        )

        subprocess.Popen([
            sys.executable,
            capture_script,
            str(student["student_id"]),
            str(student["name"])
        ])

        print(
            "Face capture started! ✅"
        )

        return redirect(
            url_for("students")
        )

    except Exception as error:

        print(
            "Capture Face Error:",
            error
        )

        return f"""
        <h2>Capture Error</h2>
        <p>{error}</p>
        """

    finally:

        if cursor:

            try:
                cursor.close()
            except:
                pass

        if connection:

            try:
                connection.close()
            except:
                pass


# ==========================================
# TRAIN MODEL
# ==========================================

@app.route("/train-model")
def train_model():

    try:

        train_script = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)
            ),
            "train.py"
        )

        if not os.path.exists(train_script):

            return """
            <h2>Training Error</h2>
            <p>train.py project folder mein nahi mila.</p>
            """

        subprocess.Popen([
            sys.executable,
            train_script
        ])

        print(
            "Face training started! 🎯"
        )

        return redirect(
            url_for("students")
        )

    except Exception as error:

        print(
            "Training Error:",
            error
        )

        return f"""
        <h2>Training Error</h2>
        <p>{error}</p>
        """


# ==========================================
# ATTENDANCE
# ==========================================

@app.route("/attendance")
def attendance():

    connection = None
    cursor = None

    try:

        search = request.args.get(
            "search",
            ""
        ).strip()

        attendance_date = request.args.get(
            "date",
            ""
        ).strip()

        status = request.args.get(
            "status",
            ""
        ).strip()

        connection = get_db_connection()

        cursor = connection.cursor(
            dictionary=True
        )

        query = """
            SELECT
                student_id,
                name,
                attendance_date,
                attendance_time,
                status
            FROM attendance
            WHERE 1 = 1
        """

        params = []

        if search:

            query += """
                AND (
                    student_id LIKE %s
                    OR name LIKE %s
                )
            """

            search_value = f"%{search}%"

            params.append(search_value)
            params.append(search_value)

        if attendance_date:

            query += """
                AND attendance_date = %s
            """

            params.append(attendance_date)

        if status:

            query += """
                AND status = %s
            """

            params.append(status)

        query += """
            ORDER BY
                attendance_date DESC,
                attendance_time DESC
        """

        cursor.execute(
            query,
            params
        )

        attendance_list = cursor.fetchall()

        total_records = len(
            attendance_list
        )

        present_records = sum(
            1
            for record in attendance_list
            if record["status"] == "Present"
        )

        absent_records = sum(
            1
            for record in attendance_list
            if record["status"] == "Absent"
        )

        return render_template(
            "attendance.html",
            attendance=attendance_list,
            total_records=total_records,
            present_records=present_records,
            absent_records=absent_records,
            search=search,
            selected_date=attendance_date,
            selected_status=status
        )

    except mysql.connector.Error as error:

        print(
            "Attendance MySQL Error:",
            error
        )

        return f"""
        <h2>Database Error</h2>
        <p>{error}</p>
        """

    finally:

        if cursor:

            try:
                cursor.close()
            except:
                pass

        if connection:

            try:
                connection.close()
            except:
                pass


# ==========================================
# REPORTS
# ==========================================

@app.route("/reports")
def reports():

    connection = None
    cursor = None

    try:

        search = request.args.get(
            "search",
            ""
        ).strip()

        from_date = request.args.get(
            "from_date",
            ""
        ).strip()

        to_date = request.args.get(
            "to_date",
            ""
        ).strip()

        connection = get_db_connection()

        cursor = connection.cursor(
            dictionary=True
        )

        query = """
            SELECT
                s.student_id,
                s.name,

                COUNT(
                    CASE
                        WHEN a.status = 'Present'
                        THEN 1
                    END
                ) AS present_count,

                COUNT(
                    CASE
                        WHEN a.status = 'Absent'
                        THEN 1
                    END
                ) AS absent_count,

                COUNT(a.student_id)
                AS total_records

            FROM students s

            LEFT JOIN attendance a
                ON s.student_id = a.student_id
        """

        params = []

        conditions = []

        if search:

            conditions.append("""
                (
                    s.student_id LIKE %s
                    OR s.name LIKE %s
                )
            """)

            search_value = f"%{search}%"

            params.append(search_value)
            params.append(search_value)

        if from_date:

            conditions.append("""
                (
                    a.attendance_date >= %s
                    OR a.attendance_date IS NULL
                )
            """)

            params.append(from_date)

        if to_date:

            conditions.append("""
                (
                    a.attendance_date <= %s
                    OR a.attendance_date IS NULL
                )
            """)

            params.append(to_date)

        if conditions:

            query += """
                WHERE
            """

            query += " AND ".join(
                conditions
            )

        query += """
            GROUP BY
                s.student_id,
                s.name

            ORDER BY
                s.name ASC
        """

        cursor.execute(
            query,
            params
        )

        report_list = cursor.fetchall()

        for report in report_list:

            total = (
                report["total_records"]
                or 0
            )

            present = (
                report["present_count"]
                or 0
            )

            if total > 0:

                report["percentage"] = round(
                    (present / total) * 100,
                    1
                )

            else:

                report["percentage"] = 0

        total_students = len(
            report_list
        )

        total_present = sum(
            report["present_count"] or 0
            for report in report_list
        )

        total_absent = sum(
            report["absent_count"] or 0
            for report in report_list
        )

        total_attendance = (
            total_present
            + total_absent
        )

        if total_attendance > 0:

            overall_percentage = round(
                (
                    total_present
                    / total_attendance
                ) * 100,
                1
            )

        else:

            overall_percentage = 0

        return render_template(
            "reports.html",

            reports=report_list,

            total_students=total_students,

            total_present=total_present,

            total_absent=total_absent,

            overall_percentage=overall_percentage,

            search=search,

            from_date=from_date,

            to_date=to_date
        )

    except mysql.connector.Error as error:

        print(
            "Reports MySQL Error:",
            error
        )

        return f"""
        <h2>Database Error</h2>
        <p>{error}</p>
        """

    finally:

        if cursor:

            try:
                cursor.close()
            except:
                pass

        if connection:

            try:
                connection.close()
            except:
                pass


# ==========================================
# EXPORT REPORT CSV
# ==========================================

@app.route("/export-report")
def export_report():

    connection = None
    cursor = None

    try:

        # --------------------------------------
        # GET SAME FILTERS AS REPORT PAGE
        # --------------------------------------

        search = request.args.get(
            "search",
            ""
        ).strip()

        from_date = request.args.get(
            "from_date",
            ""
        ).strip()

        to_date = request.args.get(
            "to_date",
            ""
        ).strip()


        # --------------------------------------
        # DATABASE
        # --------------------------------------

        connection = get_db_connection()

        cursor = connection.cursor(
            dictionary=True
        )


        # --------------------------------------
        # QUERY
        # --------------------------------------

        query = """
            SELECT
                s.student_id,
                s.name,

                COUNT(
                    CASE
                        WHEN a.status = 'Present'
                        THEN 1
                    END
                ) AS present_count,

                COUNT(
                    CASE
                        WHEN a.status = 'Absent'
                        THEN 1
                    END
                ) AS absent_count,

                COUNT(a.student_id)
                AS total_records

            FROM students s

            LEFT JOIN attendance a
                ON s.student_id = a.student_id
        """

        params = []

        conditions = []


        # --------------------------------------
        # SEARCH FILTER
        # --------------------------------------

        if search:

            conditions.append("""
                (
                    s.student_id LIKE %s
                    OR s.name LIKE %s
                )
            """)

            search_value = f"%{search}%"

            params.append(search_value)
            params.append(search_value)


        # --------------------------------------
        # FROM DATE
        # --------------------------------------

        if from_date:

            conditions.append("""
                (
                    a.attendance_date >= %s
                    OR a.attendance_date IS NULL
                )
            """)

            params.append(from_date)


        # --------------------------------------
        # TO DATE
        # --------------------------------------

        if to_date:

            conditions.append("""
                (
                    a.attendance_date <= %s
                    OR a.attendance_date IS NULL
                )
            """)

            params.append(to_date)


        if conditions:

            query += """
                WHERE
            """

            query += " AND ".join(
                conditions
            )


        query += """
            GROUP BY
                s.student_id,
                s.name

            ORDER BY
                s.name ASC
        """


        cursor.execute(
            query,
            params
        )

        rows = cursor.fetchall()


        # --------------------------------------
        # CREATE CSV IN MEMORY
        # --------------------------------------

        output = io.StringIO()

        writer = csv.writer(
            output
        )


        # --------------------------------------
        # CSV TITLE
        # --------------------------------------

        writer.writerow([
            "FaceAttend Attendance Report"
        ])

        writer.writerow([])


        # --------------------------------------
        # FILTER INFO
        # --------------------------------------

        if search:

            writer.writerow([
                "Search",
                search
            ])

        if from_date:

            writer.writerow([
                "From Date",
                from_date
            ])

        if to_date:

            writer.writerow([
                "To Date",
                to_date
            ])

        writer.writerow([])


        # --------------------------------------
        # TABLE HEADER
        # --------------------------------------

        writer.writerow([
            "Student ID",
            "Student Name",
            "Present",
            "Absent",
            "Total Records",
            "Attendance %"
        ])


        # --------------------------------------
        # DATA
        # --------------------------------------

        for row in rows:

            present = (
                row["present_count"]
                or 0
            )

            absent = (
                row["absent_count"]
                or 0
            )

            total = (
                row["total_records"]
                or 0
            )

            if total > 0:

                percentage = round(
                    (present / total) * 100,
                    1
                )

            else:

                percentage = 0


            writer.writerow([
                row["student_id"],
                row["name"],
                present,
                absent,
                total,
                f"{percentage}%"
            ])


        # --------------------------------------
        # RESPONSE
        # --------------------------------------

        csv_data = output.getvalue()

        output.close()

        filename = (
            "attendance_report.csv"
        )

        return Response(
            csv_data,
            mimetype="text/csv",
            headers={
                "Content-Disposition":
                    f"attachment; filename={filename}"
            }
        )


    except mysql.connector.Error as error:

        print(
            "Export MySQL Error:",
            error
        )

        return f"""
        <h2>Export Error</h2>
        <p>{error}</p>
        """


    except Exception as error:

        print(
            "Export Error:",
            error
        )

        return f"""
        <h2>Export Error</h2>
        <p>{error}</p>
        """


    finally:

        if cursor:

            try:
                cursor.close()
            except:
                pass

        if connection:

            try:
                connection.close()
            except:
                pass


# ==========================================
# START FACE RECOGNITION
# ==========================================

@app.route("/start-recognition")
def start_recognition():

    try:

        recognition_script = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)
            ),
            "attendance_system.py"
        )

        subprocess.Popen([
            sys.executable,
            recognition_script
        ])

        print(
            "Face recognition started! ✅"
        )

    except Exception as error:

        print(
            "Recognition Error:",
            error
        )

    return redirect(
        url_for("dashboard")
    )


# ==========================================
# RUN FLASK
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )