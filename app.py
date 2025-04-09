# --- START OF FILE app.py ---

import cv2
import os
from flask import Flask, request, render_template, redirect, url_for, flash
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import logging # Optional: For better logging

# --- Basic Logging Setup (Optional) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- Defining Flask App ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Recommended setting
# IMPORTANT: Change this secret key for production!
app.config['SECRET_KEY'] = 'replace_this_with_a_real_secret_key'

# --- Database and Extensions Setup ---
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login" # Redirect to /login if @login_required fails
login_manager.login_message_category = "info" # category for flashed message

# --- Database Models ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    # Ensure username length matches form validator max length
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)
    # Relationship to Attendance (optional but good practice)
    attendance_records = db.relationship('Attendance', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # Foreign key linking to the User table (login user)
    # Nullable=True allows recording attendance even if the face name doesn't match a login username
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    # Store the recognized name and roll from face recognition
    name = db.Column(db.String(50), nullable=False)
    roll = db.Column(db.String(50), nullable=False) # Roll might not always be Integer
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow) # Store full UTC timestamp

    def __repr__(self):
         # Format timestamp for readability if needed when printing the object
         local_time = self.timestamp.replace(tzinfo=datetime.timezone.utc).astimezone(tz=None)
         return f'<Attendance {self.name} ({self.roll}) at {local_time.strftime("%Y-%m-%d %H:%M:%S")}>'

# --- Forms ---
class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)],
                           render_kw={"placeholder": "Username", "class": "form-control"})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)],
                             render_kw={"placeholder": "Password", "class": "form-control"})
    submit = SubmitField("Register", render_kw={"class": "btn btn-primary"})

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError("Username already exists. Please choose a different one.")

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)],
                           render_kw={"placeholder": "Username", "class": "form-control"})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)],
                             render_kw={"placeholder": "Password", "class": "form-control"})
    submit = SubmitField("Login", render_kw={"class": "btn btn-primary"})

# --- Global Variables & Setup ---
N_IMAGES_PER_USER = 10 # Number of images to take for each user
DATETODAY_STR = date.today().strftime("%m_%d_%y") # For CSV filename
DATETODAY_OBJ = date.today() # Date object for DB queries

# --- Directories Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATTENDANCE_DIR = os.path.join(BASE_DIR, 'Attendance')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
FACES_DIR = os.path.join(STATIC_DIR, 'faces')
MODEL_PATH = os.path.join(STATIC_DIR, 'face_recognition_model.pkl')
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml') # Ensure this file exists

os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

# --- Initialize Attendance CSV file for the day ---
attendance_csv_file = os.path.join(ATTENDANCE_DIR, f'Attendance-{DATETODAY_STR}.csv')
if not os.path.exists(attendance_csv_file):
    try:
        with open(attendance_csv_file, 'w') as f:
            f.write('Name,Roll,Time\n') # Add header and newline
        logging.info(f"Created attendance CSV: {attendance_csv_file}")
    except IOError as e:
        logging.error(f"Could not create attendance CSV file: {e}")

# --- Load face detector ---
face_detector = None
if os.path.exists(CASCADE_PATH):
    face_detector = cv2.CascadeClassifier(CASCADE_PATH)
    if face_detector.empty():
        logging.error(f"Failed to load cascade classifier from {CASCADE_PATH}")
        face_detector = None
else:
    logging.error(f"Cascade classifier file not found at {CASCADE_PATH}")

# --- Helper Functions ---

@login_manager.user_loader
def load_user(user_id):
    """Loads user for Flask-Login."""
    return User.query.get(int(user_id))

def totalreg():
    """Returns the number of registered users (based on face folders)."""
    if not os.path.isdir(FACES_DIR):
        return 0
    # Count only directories inside FACES_DIR
    return len([d for d in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, d))])

def extract_faces(img):
    """Extract face rectangles from an image."""
    if face_detector is None:
        logging.warning("Face detector not loaded, cannot extract faces.")
        return []
    if img is None or img.size == 0:
        logging.warning("extract_faces received an invalid image.")
        return []
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Adjust parameters as needed for detection quality vs speed
        face_points = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return face_points
    except cv2.error as e:
        logging.error(f"OpenCV error during face detection: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error during face detection: {e}")
        return []

def identify_face(face_array):
    """Identify face using the trained KNN model."""
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found: {MODEL_PATH}")
        return [None]
    try:
        model = joblib.load(MODEL_PATH)
        prediction = model.predict(face_array)
        logging.info(f"Model prediction: {prediction}")
        return prediction
    except FileNotFoundError:
        logging.error(f"Model file could not be opened (might be corrupted): {MODEL_PATH}")
        return [None]
    except Exception as e:
        logging.error(f"Error loading or predicting with model: {e}")
        return [None]

def train_model():
    """Trains the KNN model on faces in static/faces."""
    faces = []
    labels = []
    user_folders = [d for d in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, d))]
    logging.info(f"Starting training with user folders: {user_folders}")

    if not user_folders:
        logging.warning("No user folders found in 'static/faces'. Training cannot proceed.")
        return False

    for user_identifier in user_folders:
        user_dir = os.path.join(FACES_DIR, user_identifier)
        img_files = [f for f in os.listdir(user_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not img_files:
             logging.warning(f"No images found for user {user_identifier} in {user_dir}. Skipping.")
             continue

        logging.info(f"Processing user: {user_identifier} with {len(img_files)} images.")
        for imgname in img_files:
            try:
                img_path = os.path.join(user_dir, imgname)
                img = cv2.imread(img_path)
                if img is None:
                    logging.warning(f"Could not read image {img_path}. Skipping.")
                    continue

                # Resize to consistent dimensions used for prediction
                resized_face = cv2.resize(img, (50, 50))
                faces.append(resized_face.ravel()) # Flatten the 50x50 image
                labels.append(user_identifier) # Label is the folder name (e.g., "John_123")
            except Exception as e:
                 logging.error(f"Error processing image {img_path}: {e}")

    if not faces or not labels:
         logging.error("No valid face data collected. Model training aborted.")
         return False

    unique_labels = len(set(labels))
    logging.info(f"Training model with {len(faces)} images and {unique_labels} unique labels.")
    faces_np = np.array(faces)

    # Ensure n_neighbors is not greater than the number of samples
    n_neighbors = min(5, len(faces_np))
    if unique_labels < n_neighbors:
        n_neighbors = unique_labels # Adjust k if fewer unique users than 5
        logging.warning(f"Adjusting n_neighbors to {n_neighbors} due to fewer unique labels.")

    if n_neighbors < 1:
        logging.error(f"Not enough unique labels ({unique_labels}) to train KNN. Need at least 1.")
        return False

    try:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(faces_np, labels)
        joblib.dump(knn, MODEL_PATH)
        logging.info(f"Model trained successfully and saved to {MODEL_PATH}")
        return True
    except Exception as e:
        logging.error(f"Error during KNN training or saving model: {e}")
        return False

def extract_attendance():
    """Extract attendance details from today's CSV file."""
    names, rolls, times = [], [], []
    count = 0
    try:
        if os.path.exists(attendance_csv_file) and os.path.getsize(attendance_csv_file) > len('Name,Roll,Time\n'): # Check size > header
             df = pd.read_csv(attendance_csv_file)
             # Ensure required columns exist
             if all(col in df.columns for col in ['Name', 'Roll', 'Time']):
                 # Convert to list to avoid issues with Pandas Series in template if empty
                 names = df['Name'].tolist()
                 rolls = df['Roll'].tolist() # Keep as object/string type from CSV
                 times = df['Time'].tolist()
                 count = len(df)
             else:
                 logging.warning(f"Attendance CSV {attendance_csv_file} missing required columns.")
        else:
             # File doesn't exist or is empty/only header
             logging.info(f"Attendance CSV {attendance_csv_file} is empty or non-existent.")
             # Recreate header if file exists but is empty
             if os.path.exists(attendance_csv_file) and os.path.getsize(attendance_csv_file) == 0:
                  with open(attendance_csv_file, 'w') as f: f.write('Name,Roll,Time\n')

    except pd.errors.EmptyDataError:
         logging.warning(f"Attendance CSV {attendance_csv_file} is empty.")
         # File might exist but be empty, ensure header
         if os.path.exists(attendance_csv_file):
             try:
                 with open(attendance_csv_file, 'w') as f: f.write('Name,Roll,Time\n')
             except IOError as e:
                 logging.error(f"Could not write header to empty CSV file: {e}")
    except Exception as e:
        logging.error(f"Error reading attendance CSV {attendance_csv_file}: {e}")

    logging.info(f"Extracted {count} records from CSV.")
    return names, rolls, times, count


def add_attendance(name, roll_str):
    """Adds attendance record to CSV and Database if not already present for the day."""
    logging.info(f"Attempting to add attendance for: Name='{name}', Roll='{roll_str}'")
    current_time_str = datetime.now().strftime("%H:%M:%S")
    current_timestamp_utc = datetime.utcnow()
    added_csv = False
    added_db = False

    # --- CSV Logging ---
    try:
        # Read existing data, treating Roll as string for comparison
        try:
             df = pd.read_csv(attendance_csv_file)
             existing_rolls = df['Roll'].astype(str).tolist()
        except (FileNotFoundError, pd.errors.EmptyDataError):
             df = pd.DataFrame(columns=['Name', 'Roll', 'Time']) # Create empty df if file missing/empty
             existing_rolls = []

        if roll_str not in existing_rolls:
            new_entry = pd.DataFrame([[name, roll_str, current_time_str]], columns=['Name', 'Roll', 'Time'])
            # Append using pandas or write directly
            try:
                with open(attendance_csv_file, 'a', newline='') as f:
                    # Avoid writing header again if file already exists
                    f.write(f"{name},{roll_str},{current_time_str}\n")
                logging.info(f"Attendance added to CSV for {name} ({roll_str})")
                added_csv = True
            except IOError as e:
                 logging.error(f"IOError writing to CSV {attendance_csv_file}: {e}")

        else:
            logging.info(f"Attendance already recorded in CSV today for Roll: {roll_str}")

    except Exception as e:
        logging.error(f"Error processing CSV file {attendance_csv_file}: {e}")

    # --- Database Logging ---
    try:
        # Attempt to find the corresponding User (login user) based on the 'name' part.
        # Assumes login username == face name used during registration.
        user = User.query.filter_by(username=name).first()
        db_user_id = user.id if user else None
        if user:
             logging.info(f"Found matching login user: ID={db_user_id}")
        else:
             logging.warning(f"No login user found with username '{name}'. Attendance DB record will have null user_id.")

        # Check if a record for this person (name, roll) exists for today in the DB
        start_of_day_utc = datetime.combine(DATETODAY_OBJ, datetime.min.time(), tzinfo=datetime.timezone.utc)
        end_of_day_utc = datetime.combine(DATETODAY_OBJ, datetime.max.time(), tzinfo=datetime.timezone.utc)

        existing_attendance = Attendance.query.filter(
            Attendance.name == name,
            Attendance.roll == roll_str,
            Attendance.timestamp >= start_of_day_utc,
            Attendance.timestamp <= end_of_day_utc
        ).first()

        if not existing_attendance:
            try:
                attendance_record = Attendance(
                    user_id=db_user_id,
                    name=name,
                    roll=roll_str,
                    timestamp=current_timestamp_utc
                )
                db.session.add(attendance_record)
                db.session.commit()
                logging.info(f"Attendance added to DB for {name} ({roll_str})")
                added_db = True
            except Exception as e:
                db.session.rollback()
                logging.error(f"Error adding attendance to database: {e}")
        else:
            logging.info(f"Attendance already recorded in DB today for {name} ({roll_str})")

    except Exception as e:
        logging.error(f"Error during database attendance check/add: {e}")

    return added_csv or added_db # Return True if added to either

# --- Create Database Tables ---
# This ensures tables are created based on models before the first request
# In production, consider using Flask-Migrate for schema changes
with app.app_context():
    try:
        db.create_all()
        logging.info("Database tables checked/created.")
    except Exception as e:
        logging.error(f"Error creating database tables: {e}")


# --- Flask Routes ---

@app.route('/')
def home():
    """Render the home page (simple landing or redirect)."""
    # Could redirect to login or dashboard if logged in
    # return render_template('home.html') # If you have a home template
    if current_user.is_authenticated:
         return redirect(url_for('dashboard'))
    return redirect(url_for('login')) # Or render a specific home page


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            logging.info(f"User '{user.username}' logged in successfully.")
            # Redirect to the page they were trying to access, or dashboard
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')
            logging.warning(f"Failed login attempt for username: {form.username.data}")
    return render_template('login.html', form=form, title="Login")


@app.route('/dashboard')
@login_required
def dashboard():
    """Display the main dashboard after login."""
    user_details = current_user # Get the full user object via current_user proxy
    names, rolls, times, count = extract_attendance()
    # Fetch attendance from DB for today (optional, more robust)
    # start_of_day_utc = datetime.combine(DATETODAY_OBJ, datetime.min.time(), tzinfo=datetime.timezone.utc)
    # end_of_day_utc = datetime.combine(DATETODAY_OBJ, datetime.max.time(), tzinfo=datetime.timezone.utc)
    # db_records = Attendance.query.filter(Attendance.timestamp >= start_of_day_utc, Attendance.timestamp <= end_of_day_utc).order_by(Attendance.timestamp).all()
    # Pass db_records to template if using DB data primarily

    return render_template('dashboard.html',
                           names=names, rolls=rolls, times=times, l=count,
                           totalreg=totalreg(), user_details=user_details, title="Dashboard")


@app.route('/logout')
@login_required
def logout():
    """Handle user logout."""
    logging.info(f"User '{current_user.username}' logging out.")
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle new user registration (for login system)."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        new_user = User(username=form.username.data, password=hashed_password)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash(f'Account created for {form.username.data}! You can now log in.', 'success')
            logging.info(f"New user registered: {form.username.data}")
            return redirect(url_for('login'))
        except Exception as e:
             db.session.rollback()
             logging.error(f"Error during registration for {form.username.data}: {e}")
             flash('Registration failed due to a server error. Please try again.', 'danger')
    return render_template('register.html', form=form, title="Register")


# This route might be redundant if dashboard shows the same info
@app.route('/attendance_list')
@login_required
def attendance_list():
    """Display the attendance list (primarily from CSV)."""
    names, rolls, times, count = extract_attendance()
    return render_template('attendance_list.html', # Use a specific template if needed
                           names=names, rolls=rolls, times=times, l=count,
                           totalreg=totalreg(), title="Attendance List")


@app.route('/start', methods=['GET'])
@login_required
def start():
    """Start the face recognition process to take attendance."""
    if face_detector is None:
        flash('Face detector not available. Cannot start attendance.', 'danger')
        logging.error("Attempted to start attendance but face detector is not loaded.")
        return redirect(url_for('dashboard'))

    if not os.path.exists(MODEL_PATH):
        flash('Trained model not found. Please add users and train the model first.', 'warning')
        logging.warning("Attempted to start attendance but model file is missing.")
        # Redirect instead of rendering template here to avoid showing empty dashboard
        return redirect(url_for('dashboard'))

    logging.info("Starting attendance check via webcam...")
    cap = cv2.VideoCapture(0) # Use 0 for default webcam
    if not cap.isOpened():
        flash('Error: Could not open webcam.', 'danger')
        logging.error("Failed to open webcam (index 0).")
        return redirect(url_for('dashboard'))

    recognition_active = True
    processed_today = set() # Keep track of rolls processed in this session

    while recognition_active:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame from webcam during attendance.")
            flash("Error capturing frame from webcam.", "danger")
            break # Exit loop if frame capture fails

        faces = extract_faces(frame)

        display_frame = frame.copy() # Work on a copy to draw on

        if len(faces) > 0:
            # Process only the first detected face for simplicity
            (x, y, w, h) = faces[0]

            # Ensure coordinates are valid and within frame boundaries
            fh, fw = display_frame.shape[:2]
            x1, y1, x2, y2 = max(0, x), max(0, y), min(fw, x+w), min(fh, y+h)
            if x2 <= x1 or y2 <= y1: # Check for valid width/height
                continue

            # Crop and resize face
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                 logging.warning("Cropped face image is empty.")
                 continue

            try:
                resized_face = cv2.resize(face_img, (50, 50))
                face_array = resized_face.reshape(1, -1)
            except cv2.error as e:
                 logging.error(f"OpenCV error resizing face: {e}")
                 continue # Skip this frame

            # Identify face
            prediction_list = identify_face(face_array)
            identified_person = prediction_list[0] if prediction_list and prediction_list[0] is not None else "Unknown"

            display_text = "Unknown"
            color = (0, 0, 255) # Red for Unknown/Error

            if identified_person != "Unknown":
                try:
                    # Parse the name and roll from the identifier (e.g., "John_123")
                    name_part, roll_part = identified_person.split('_', 1)
                    display_text = name_part # Display only the name part

                    # Check if already processed in this session to avoid spamming logs/DB attempts
                    if roll_part not in processed_today:
                         if add_attendance(name_part, roll_part):
                              # Successfully added (either CSV or DB)
                              color = (0, 255, 0) # Green for success
                              processed_today.add(roll_part) # Mark as processed for this session
                              flash(f"Attendance marked for {name_part} ({roll_part})", "success") # Give feedback
                         else:
                              # Already recorded today (or error during add)
                              color = (0, 255, 255) # Yellow for already marked
                              display_text += " (Marked)"
                              # No need to add to processed_today if add_attendance returned false because already present
                    else:
                        # Already processed in *this specific run*
                        color = (0, 255, 255) # Yellow
                        display_text += " (Marked)"

                except ValueError:
                    logging.error(f"Could not parse identifier '{identified_person}'. Expected 'Name_Roll'.")
                    display_text = "ID Format Error"
                    color = (0, 0, 255)
                except Exception as e:
                     logging.error(f"Error during attendance addition call for {identified_person}: {e}")
                     display_text = "Add Error"
                     color = (0, 0, 255)
            else:
                 # identified_person is "Unknown"
                 pass # Keep text as Unknown and color red

            # Draw rectangle around face
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            # Draw text background
            cv2.rectangle(display_frame, (x1, y1 - 35), (x2, y1), color, -1)
            # Draw text
            cv2.putText(display_frame, display_text, (x1 + 6, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the frame
        try:
            cv2.imshow('Attendance Check - Press ESC to Exit', display_frame)
        except cv2.error as e:
             logging.error(f"cv2.imshow error: {e}")
             # This can happen if the window is closed unexpectedly
             recognition_active = False # Try to exit gracefully

        # Exit condition
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # 27 is the ASCII code for the ESC key
            logging.info("ESC key pressed. Stopping attendance check.")
            recognition_active = False
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    # Attempt to destroy windows again, sometimes needed if closed manually
    for i in range(5):
        cv2.waitKey(1)

    logging.info("Attendance check process finished.")
    # Redirect back to dashboard, which will fetch updated attendance
    return redirect(url_for('dashboard'))


@app.route('/add', methods=['POST'])
@login_required
def add():
    """Add a new user's face data and retrain the model."""
    if face_detector is None:
        flash('Face detector not available. Cannot add user faces.', 'danger')
        return redirect(url_for('dashboard'))

    try:
        newusername = request.form['newusername'].strip()
        newuserid = request.form['newuserid'].strip() # This is the "roll number"

        if not newusername or not newuserid:
             flash('Username and User ID (Roll) are required.', 'warning')
             return redirect(url_for('dashboard'))

        # Basic validation (e.g., no weird characters) could be added here
        # Replace spaces or problematic chars if needed
        safe_username = "".join(c if c.isalnum() else "_" for c in newusername)
        safe_userid = "".join(c if c.isalnum() else "_" for c in newuserid)

        # Use username_userid as the unique identifier/folder name
        user_identifier = f"{safe_username}_{safe_userid}"
        user_folder = os.path.join(FACES_DIR, user_identifier)

        if os.path.exists(user_folder):
             flash(f'User "{newusername}" with ID "{newuserid}" already exists. To add more images, please remove the existing folder first.', 'warning')
             logging.warning(f"Attempted to add existing user: {user_identifier}")
             return redirect(url_for('dashboard'))
        else:
            try:
                os.makedirs(user_folder)
                logging.info(f"Created folder for new user: {user_folder}")
            except OSError as e:
                 flash(f"Error creating directory for user: {e}", "danger")
                 logging.error(f"OSError creating folder {user_folder}: {e}")
                 return redirect(url_for('dashboard'))

        # --- Start Webcam Capture ---
        images_captured_count = 0
        frame_counter = 0

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            flash("Error: Could not open webcam.", "danger")
            logging.error("Failed to open webcam for adding user.")
            # Clean up created folder if webcam fails
            try:
                if os.path.exists(user_folder): os.rmdir(user_folder)
            except OSError: pass
            return redirect(url_for('dashboard'))

        logging.info(f"Starting face capture for {user_identifier}")
        capture_active = True
        while capture_active and images_captured_count < N_IMAGES_PER_USER:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture frame during face addition.")
                flash("Error capturing frame from webcam.", "danger")
                break

            faces = extract_faces(frame)
            display_frame = frame.copy()

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                x1, y1, x2, y2 = max(0, x), max(0, y), min(frame.shape[1], x+w), min(frame.shape[0], y+h)

                # Draw rectangle
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 20), 2)

                # Capture an image every few frames (e.g., every 5th frame)
                # Add a small delay initially (e.g., j > 10) to allow user positioning
                if frame_counter > 10 and frame_counter % 5 == 0:
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size > 0:
                        img_name = f'{user_identifier}_{images_captured_count}.jpg'
                        img_path = os.path.join(user_folder, img_name)
                        try:
                            cv2.imwrite(img_path, face_img)
                            logging.info(f"Saved image: {img_path}")
                            images_captured_count += 1
                        except cv2.error as e:
                             logging.error(f"OpenCV error saving image {img_path}: {e}")
                        except Exception as e:
                             logging.error(f"Error saving image {img_path}: {e}")

                frame_counter += 1

                # Display capture progress text
                progress_text = f'Images Captured: {images_captured_count}/{N_IMAGES_PER_USER}'
                cv2.putText(display_frame, progress_text, (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 20), 2, cv2.LINE_AA)

            # Display frame
            try:
                cv2.imshow(f'Adding Face: {newusername} ({newuserid}) - Press ESC to Cancel', display_frame)
            except cv2.error as e:
                logging.error(f"cv2.imshow error during add face: {e}")
                capture_active = False # Exit loop

            # Exit condition
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC key
                logging.info("ESC key pressed during face capture. Cancelling.")
                capture_active = False
                break # Exit capture loop

        # --- Cleanup Webcam ---
        cap.release()
        cv2.destroyAllWindows()
        for i in range(5): cv2.waitKey(1) # Ensure windows are closed

        # --- Process Results ---
        if images_captured_count < N_IMAGES_PER_USER:
             # Capture was cancelled or failed before completion
             flash(f'Face capture cancelled or incomplete ({images_captured_count}/{N_IMAGES_PER_USER} images saved). User not fully added.', 'warning')
             logging.warning(f"Face capture for {user_identifier} incomplete. Removing folder.")
             # Clean up partially created folder
             try:
                 if os.path.exists(user_folder):
                     for filename in os.listdir(user_folder):
                         os.remove(os.path.join(user_folder, filename))
                     os.rmdir(user_folder)
             except OSError as e:
                 logging.error(f"Error removing incomplete user folder {user_folder}: {e}")
        else:
            # Capture completed successfully
            flash(f'Successfully captured {images_captured_count} images for {newusername}. Now training model...', 'info')
            logging.info(f"Completed face capture for {user_identifier}. Training model...")
            if train_model():
                flash('Model training successful!', 'success')
                logging.info("Model training successful after adding user.")
            else:
                 flash('User images saved, but model training failed. Please check logs.', 'danger')
                 logging.error("Model training failed after adding user.")

    except KeyError:
        flash('Form data missing (Username or User ID).', 'danger')
        logging.error("KeyError accessing form data in /add route.")
    except Exception as e:
         flash(f'An unexpected error occurred: {e}', 'danger')
         logging.error(f"An unexpected error occurred in /add route: {e}", exc_info=True) # Log traceback

    # Redirect back to dashboard regardless of outcome
    return redirect(url_for('dashboard'))

# --- Main Execution ---
if __name__ == '__main__':
    # Use debug=False for production
    # Use host='0.0.0.0' to make accessible on local network
    app.run(debug=True, host='0.0.0.0')
# --- END OF FILE app.py ---