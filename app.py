from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os 
import imghdr
from datetime import datetime


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id  
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('auth.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        


        if User.query.filter_by(email=email).first():
            flash('Email address already in use. Please choose a different one.', 'danger')
            return render_template('auth.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('auth.html')

        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return render_template('auth.html')

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('auth.html')

@app.route('/home')
def home():
    return render_template('home.html')

model = YOLO('best.pt')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        uploaded_file = request.files['image_file']

        if uploaded_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)
            image = cv2.imread(file_path)
            results = model(image)
            result = results[0]  
            
            confidence_threshold = 0.4
            
            detected_classes = []  
            detection_details = []  
            
            # Process all detections
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xywh.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                
                for i, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    class_id = int(class_id)
                    confidence = float(confidence)
                    label_name = result.names[class_id]
                    
                    if confidence >= confidence_threshold:
                        detection_info = {
                            'class': label_name,
                            'confidence': confidence,
                            'confidence_percentage': f"{confidence:.2%}",
                            'box_index': i
                        }
                        
                        detected_classes.append(detection_info)
                        detection_details.append(detection_info)
                        
                        x_center, y_center, width, height = box  
                        x1 = int((x_center - width / 2))
                        y1 = int((y_center - height / 2))
                        x2 = int((x_center + width / 2))
                        y2 = int((y_center + height / 2))

                        color = (0, 255, 0) if label_name == 'Nonanemic' else (0, 0, 255)
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2) 

                        label_text = f"{label_name} {confidence:.3f}"
                        label_position = (x1, y1 - 10) if y1 > 20 else (x1, y1 + 20)

                        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(image, (x1, y1-25), (x1 + w + 5, y1), color, -1)
                        
                        cv2.putText(image, label_text, (x1 + 2, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + uploaded_file.filename)
            cv2.imwrite(result_image_path, image)

            for det in detection_details:
                print(f"Box {det['box_index']}: {det['class']}, Confidence: {det['confidence']:.4f} ({det['confidence_percentage']})")
            
            confidence_reason = "Baseline classification"
            final_confidence = 0.0
            
            if detected_classes:
                best_detection = max(detected_classes, key=lambda x: x['confidence'])
                highest_confidence = best_detection['confidence']
                
                if highest_confidence < 0.30:
                    final_class = 'Anemic'
                    confidence_reason = f"confidence ({highest_confidence:.2%} < 30%)"
                else:
                    final_class = 'Nonanemic' 
                    confidence_reason = f"confidence ({highest_confidence:.2%} >= 30%)"
                
                final_confidence = highest_confidence
            else:
                print("No detections above threshold - using default classification: Nonanemic")
            
            print("=== End of Results ===\n")

            if detected_classes:
                all_detections_msg = " | ".join([f"{det['class']} ({det['confidence_percentage']})" for det in detected_classes])
                detection_message = f"Detections: {all_detections_msg} "
            else:
                detection_message = f"Result: {final_class} | {confidence_reason}"
            return render_template('upload.html', 
                                 result_image=url_for('uploaded_file', filename='result_' + uploaded_file.filename), 
                                 detection_message=detection_message)

    return render_template('upload.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
