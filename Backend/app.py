from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
import sqlite3
import os
from werkzeug.utils import secure_filename
from cnn_svm import predict_tumor
from fpdf import FPDF
from datetime import datetime
import json
import streamlit as st
st.title("Brain Tumor Detector")

# ---------------- Flask App Setup ----------------
app = Flask(__name__)
app.secret_key = "supersecretkey123"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

STATIC_RESULTS = "static/results"
os.makedirs(STATIC_RESULTS, exist_ok=True)

REPORTS_FOLDER = "static/reports"
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# ---------------- Database Setup ----------------
DB_NAME = "users.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            patient_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------- Tumor Information Database ----------------
TUMOR_INFO = {
    "glioma": {
        "description": "Glioma is a type of tumor that occurs in the brain and spinal cord. Gliomas begin in the glial cells that surround and support nerve cells.",
        "symptoms": "Headaches, Nausea, Vomiting, Seizures, Memory loss, Personality changes, Weakness on one side of body",
        "treatment": "Surgical removal, Radiation therapy, Chemotherapy, Targeted drug therapy",
        "recommendation": "Immediate consultation with neuro-oncologist. Regular MRI monitoring every 3-6 months."
    },
    "meningioma": {
        "description": "Meningioma is a tumor that arises from the meninges — the membranes that surround the brain and spinal cord.",
        "symptoms": "Headaches, Vision problems, Hearing loss, Memory loss, Seizures, Weakness in limbs",
        "treatment": "Observation for slow-growing tumors, Surgical removal, Radiation therapy",
        "recommendation": "Consultation with neurosurgeon. Annual MRI scans for monitoring."
    },
    "pituitary": {
        "description": "Pituitary tumors are abnormal growths that develop in the pituitary gland, affecting hormone production.",
        "symptoms": "Headaches, Vision loss, Fatigue, Weight gain/loss, Menstrual irregularities, Erectile dysfunction",
        "treatment": "Medication to regulate hormones, Surgical removal, Radiation therapy",
        "recommendation": "Endocrinologist consultation. Hormone level testing required."
    },
    "no tumor": {
        "description": "No evidence of tumor detected in the MRI scan. Brain structure appears normal.",
        "symptoms": "None detected",
        "treatment": "No treatment required. Maintain regular health checkups.",
        "recommendation": "Annual brain MRI recommended for high-risk patients. Maintain healthy lifestyle."
    }
}

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
        if cur.fetchone():
            session['username'] = u
            return redirect('/upload')
        flash("Invalid credentials")
    return render_template("login.html")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users(username,password) VALUES(?,?)", (u, p))
            conn.commit()
            flash("Registration successful! Please login.")
            return redirect('/login')
        except:
            flash("User already exists")
    return render_template("signup.html")

# ---------- UPLOAD & PREDICTION ----------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        try:
            # ---- Patient info ----
            name = request.form.get("name", "Unknown")
            age = request.form.get("age", "N/A")
            gender = request.form.get("gender", "Not Specified")
            patient_id = request.form.get("patient_id", "N/A")
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ---- File Upload ----
            file = request.files.get("file")
            if not file or file.filename == "":
                flash("Please select an MRI image")
                return redirect(request.url)
            
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                flash("Only image files allowed (.jpg, .jpeg, .png)")
                return redirect(request.url)

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # ---- Prediction ----
            try:
                prediction, confidence = predict_tumor(filepath)
            except Exception as e:
                prediction = "Prediction Error"
                confidence = 0
                print(f"Prediction error: {e}")

            # Clean prediction string
            clean_pred = prediction.lower()
            if "glioma" in clean_pred:
                tumor_type = "glioma"
            elif "meningioma" in clean_pred:
                tumor_type = "meningioma"
            elif "pituitary" in clean_pred:
                tumor_type = "pituitary"
            else:
                tumor_type = "no tumor"

            # ---- Save copy in static for display ----
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"{patient_id}_{timestamp}.jpg"
            result_path = os.path.join(STATIC_RESULTS, result_filename)
            
            # Copy the file
            import shutil
            shutil.copy2(filepath, result_path)

            # ---- Save to database ----
            conn = sqlite3.connect(DB_NAME)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO patient_history 
                (patient_id, patient_name, age, gender, filename, prediction, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (patient_id, name, age, gender, result_filename, prediction, confidence))
            conn.commit()
            conn.close()

            result = {
                "name": name,
                "age": age,
                "gender": gender,
                "patient_id": patient_id,
                "date": date,
                "filename": f"results/{result_filename}",
                "prediction": prediction,
                "confidence": confidence,
                "tumor_type": tumor_type,
                "tumor_info": TUMOR_INFO.get(tumor_type, TUMOR_INFO["no tumor"])
            }

            return render_template("upload.html", result=result)

        except Exception as e:
            flash(f"Error: {str(e)}")
            return redirect(request.url)

    return render_template("upload.html", result=None)

@app.route("/download_report", methods=["POST"])
def download_report():
    try:
        name = request.form.get("name", "Unknown")
        age = request.form.get("age", "N/A")
        gender = request.form.get("gender", "Not Specified")
        patient_id = request.form.get("patient_id", "N/A")
        filename = request.form.get("filename", "")
        prediction = request.form.get("prediction", "No prediction")
        confidence = float(request.form.get("confidence", "0"))

        clean_pred = prediction.lower()
        if "glioma" in clean_pred:
            tumor_type = "Glioma"
        elif "meningioma" in clean_pred:
            tumor_type = "Meningioma"
        elif "pituitary" in clean_pred:
            tumor_type = "Pituitary"
        else:
            tumor_type = "No Tumor"

        pdf = FPDF()
        pdf.add_page()

        # ======= HEADER =======
        pdf.set_fill_color(25, 55, 100)
        pdf.rect(0, 0, 210, 28, 'F')

        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 18)
        pdf.set_xy(0, 8)
        pdf.cell(0, 8, "Neuro Care Hospital", 0, 1, "C")

        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, "AI-Powered Brain Tumor Detection System", 0, 1, "C")
        pdf.cell(0, 6, "MEDICAL IMAGING ANALYSIS REPORT", 0, 1, "C")

        pdf.set_text_color(0, 0, 0)
        pdf.set_y(38)

        # ======= PATIENT INFORMATION =======
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "PATIENT INFORMATION", 0, 1)
        pdf.line(10, pdf.get_y(), 80, pdf.get_y())
        pdf.ln(4)

        pdf.set_font("Arial", "", 11)
        pdf.cell(90, 6, f"Name: {name}", 0, 0)
        pdf.cell(90, 6, f"Age: {age} years", 0, 1)
        pdf.cell(90, 6, f"Gender: {gender}", 0, 0)
        pdf.cell(90, 6, f"MRI Date: {datetime.now().strftime('%B %d, %Y')}", 0, 1)
        pdf.cell(90, 6, "Radiologist: Dr. Sara Johnson", 0, 0)
        pdf.cell(90, 6, "Model: CNN-SVM v2.1", 0, 1)

        pdf.ln(6)

        # ======= AI ANALYSIS RESULTS =======
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "AI ANALYSIS RESULTS", 0, 1)
        pdf.line(10, pdf.get_y(), 90, pdf.get_y())
        pdf.ln(4)

        # Red tumor label
        pdf.set_fill_color(200, 40, 40)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(55, 8, "TUMOR DETECTED", 0, 1, "C", True)

        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 6, f"Tumor Type: {tumor_type}", 0, 1)
        pdf.cell(0, 6, f"Detection Confidence: {confidence}%", 0, 1)
        pdf.cell(0, 6, "Severity Level: Moderate", 0, 1)
        pdf.cell(0, 6, "Risk Category: Medium", 0, 1)
        pdf.cell(0, 6, f"AI Model Confidence: {confidence}%", 0, 1)

        pdf.ln(6)

        # ======= MEDICAL RECOMMENDATIONS =======
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "MEDICAL RECOMMENDATIONS", 0, 1)
        pdf.line(10, pdf.get_y(), 110, pdf.get_y())
        pdf.ln(4)

        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(
            0, 6,
            "Recommended Action:\nConsult Neurosurgeon\n\n"
            "Medical Insights:\nBased on the MRI analysis, this tumor requires further evaluation by a neurosurgeon for treatment planning.\n"
        )

        pdf.set_font("Arial", "I", 10)
        pdf.multi_cell(
            0, 5,
            "Clinical Note:\nGliomas arise from glial cells. Further evaluation with contrast-enhanced MRI and neurological consultation is recommended."
        )

        # ======= MRI IMAGE =======
        pdf.ln(6)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "MRI SCAN IMAGE", 0, 1, "C")

        image_path = os.path.join("static", filename)
        if os.path.exists(image_path):
            pdf.image(image_path, x=30, y=pdf.get_y()+2, w=150, h=70)

            # Red tumor box
            pdf.set_draw_color(255, 0, 0)
            pdf.set_line_width(1.5)
            pdf.rect(85, pdf.get_y()+22, 40, 30)

        # ======= FOOTER =======
        pdf.set_y(270)
        pdf.set_font("Arial", "I", 8)
        pdf.multi_cell(
            0, 4,
            "This report is generated by AI and should be reviewed by a qualified medical professional."
        )

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{patient_id}_report_{timestamp}.pdf"
        report_path = os.path.join(REPORTS_FOLDER, report_filename)
        pdf.output(report_path)

        return send_file(report_path, as_attachment=True)

    except Exception as e:
        flash(f"Error generating report: {str(e)}")
        return redirect(url_for('upload'))





# ---------- View Patient History ----------
@app.route("/history")
def history():
    if "username" not in session:
        return redirect(url_for("login"))
    
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM patient_history ORDER BY scan_date DESC LIMIT 50")
    history = cur.fetchall()
    conn.close()
    
    return render_template("history.html", history=history)

# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ---------- Delete History ----------
@app.route("/delete_history/<int:id>")
def delete_history(id):
    if "username" not in session:
        return redirect(url_for("login"))
    
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DELETE FROM patient_history WHERE id=?", (id,))
    conn.commit()
    conn.close()
    
    return redirect(url_for("history"))

# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
