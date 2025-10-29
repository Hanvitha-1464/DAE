import os
import sqlite3
import numpy as np
import re
from flask import Flask, render_template, request, redirect, url_for, session, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Image quality metrics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

# Flask setup
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database setup
DB_PATH = 'database/users.db'
os.makedirs('database', exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )''')
    conn.commit()
    conn.close()
init_db()


def add_user(name, email, username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    hashed_pw = generate_password_hash(password)
    try:
        c.execute("INSERT INTO users (name, email, username, password) VALUES (?,?,?,?)",
                  (name, email, username, hashed_pw))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return False
    conn.close()
    return True


def verify_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row and check_password_hash(row[0], password):
        return True
    return False


def validate_username(username):
    if not username or len(username) < 3 or len(username) > 20:
        return False, "Username must be 3-20 characters"
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores"
    return True, ""


def validate_password(password):
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    return True, ""


def validate_email(email):
    if not email or len(email) < 5:
        return False, "Email is required"
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return False, "Invalid email format"
    return True, ""


def validate_name(name):
    if not name or len(name.strip()) < 2:
        return False, "Name must be at least 2 characters"
    if len(name) > 50:
        return False, "Name must be less than 50 characters"
    return True, ""


def validate_image(file):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    MAX_SIZE = 5 * 1024 * 1024  # 5MB

    if not file or file.filename == '':
        return False, "No file selected"

    ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Only {', '.join(ALLOWED_EXTENSIONS)} files allowed"

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_SIZE:
        return False, "File size must be less than 5MB"

    return True, ""


# Load ML model
MODEL_PATH = os.path.join('model', 'denoising_autoencoder_cnn.keras')
model = load_model(MODEL_PATH)


def preprocess_image_pil(img: Image.Image, size=(256,256)):
    img = img.convert('RGB')
    img = img.resize(size, Image.BICUBIC)
    arr = img_to_array(img).astype('float32') / 255.0
    return np.expand_dims(arr, axis=0)


def denoise_and_save(in_path, out_name='denoised.jpg'):
    """Returns: out_path (str), metrics (dict with PSNR, MSE, SSIM)"""
    img = Image.open(in_path).convert('RGB')
    original_size = img.size

    # Process through model
    input_arr = preprocess_image_pil(img, size=(256,256))
    decoded = model.predict(input_arr)
    decoded = np.clip(decoded[0], 0.0, 1.0)
    decoded_img = Image.fromarray((decoded * 255).astype('uint8'))

    # Resize back for saving
    decoded_img_resized = decoded_img.resize(original_size, Image.LANCZOS)
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
    decoded_img_resized.save(out_path)

    # --- Compute metrics ---
    orig_resized = img.resize((256,256), Image.BICUBIC)
    recon_resized = decoded_img.resize((256,256), Image.BICUBIC)
    orig_arr = np.array(orig_resized).astype('float32') / 255.0
    recon_arr = np.array(recon_resized).astype('float32') / 255.0

    try:
        psnr_val = peak_signal_noise_ratio(orig_arr, recon_arr, data_range=1.0)
    except Exception:
        psnr_val = float('nan')
    try:
        mse_val = mean_squared_error(orig_arr, recon_arr)
    except Exception:
        mse_val = float('nan')
    try:
        ssim_val = structural_similarity(orig_arr, recon_arr, channel_axis=2, data_range=1.0)
    except Exception:
        ssim_val = float('nan')

    metrics = {
        "PSNR": round(float(psnr_val), 3) if not np.isnan(psnr_val) else "N/A",
        "MSE": round(float(mse_val), 6) if not np.isnan(mse_val) else "N/A",
        "SSIM": round(float(ssim_val), 4) if not np.isnan(ssim_val) else "N/A"
    }

    return out_path, metrics


# Routes
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('denoise'))
    return render_template('homepage.html')


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if verify_user(username, password):
            session['username'] = username
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT name FROM users WHERE username=?", (username,))
            row = c.fetchone()
            conn.close()
            if row:
                session['name'] = row[0]
            return redirect(url_for('denoise'))
        else:
            flash("Invalid credentials", "danger")
    return render_template('login.html')


@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name'].strip()
        email = request.form['email'].strip().lower()
        username = request.form['username'].strip()
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        valid, msg = validate_name(name)
        if not valid:
            flash(msg, "danger")
            return render_template('signup.html')

        valid, msg = validate_email(email)
        if not valid:
            flash(msg, "danger")
            return render_template('signup.html')

        valid, msg = validate_username(username)
        if not valid:
            flash(msg, "danger")
            return render_template('signup.html')

        valid, msg = validate_password(password)
        if not valid:
            flash(msg, "danger")
            return render_template('signup.html')

        if password != confirm_password:
            flash("Passwords do not match", "danger")
            return render_template('signup.html')

        success = add_user(name, email, username, password)
        if success:
            flash("Signup successful. Please login.", "success")
            return redirect(url_for('login'))
        else:
            flash("Username or email already exists!", "danger")
    return render_template('signup.html')


@app.route('/denoise', methods=['GET','POST'])
def denoise():
    if 'username' not in session:
        return redirect(url_for('login'))

    uploaded_url = None
    output_url = None
    metrics = None

    if request.method == 'POST':
        f = request.files.get('file')

        valid, msg = validate_image(f)
        if not valid:
            flash(msg, "danger")
            return render_template('index.html', uploaded_image=None, output_image=None, metrics=None)

        fname = secure_filename(f.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        f.save(save_path)

        out_path, metrics = denoise_and_save(save_path, out_name='denoised_' + fname)
        uploaded_url = url_for('static', filename=f'uploads/{fname}')
        output_url = url_for('static', filename=f'uploads/denoised_{fname}')

    return render_template('index.html',
                           uploaded_image=uploaded_url,
                           output_image=output_url,
                           metrics=metrics)


@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('name', None)
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
