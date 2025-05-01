import os
from flask import Flask, request, render_template, flash, redirect, url_for, session
import bcrypt
from werkzeug.utils import secure_filename
import torch
import numpy as np
from model import SiameseCNN1D
from process_video import process_video_keypoints
from sklearn.metrics.pairwise import euclidean_distances
import random
import logging

USER_FILE = 'users.txt'  # File untuk menyimpan data pengguna

# Fungsi untuk memuat pengguna dari file
def load_users():
    users = {}
    if os.path.exists(USER_FILE):
        with open(USER_FILE, 'r') as f:
            for line in f:
                email, username, hashed_password = line.strip().split(',')
                users[email] = {'username': username, 'password': hashed_password}
    return users

# Fungsi untuk menyimpan pengguna ke file
def save_user(email, username, hashed_password):
    with open(USER_FILE, 'a') as f:
        f.write(f"{email},{username},{hashed_password}\n")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'C:\Users\bcamaster\Documents\Ardo\AIR-Handsign\Web\fsl_video_detection\static\uploads'
app.config['SUPPORT_SET_DIR'] = r"C:\Users\bcamaster\Documents\Ardo\AIR-Handsign\Web\fsl_video_detection\support_set"
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Batas ukuran file 50MB
app.secret_key = "super_secret_key"

# Pastikan folder upload dan support set ada
for folder in [app.config['UPLOAD_FOLDER'], app.config['SUPPORT_SET_DIR']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Muat model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseCNN1D(input_channels=84, embedding_size=128)  # Inisialisasi model
try:
    model.load_state_dict(
        torch.load(
            r"C:\Users\bcamaster\Documents\Ardo\AIR-Handsign\Web\fsl_video_detection\models\siamese_cnn1d_1layer_mediapipe_KataHurufAngka_hardtriplet_margin2_epoch300_try9.pth",
            map_location=device,
            weights_only=True
        )
    )
except Exception as e:
    print(f"Error loading model weights: {e}")
    raise
model.to(device)
model.eval()

# Muat support set dengan batasan jumlah per kelas
support_embeddings = []
support_labels = []
classes = ["Aku", "Semangat", "Cinta", "Kenapa", "Tidak", "Kecewa", "A", "B", "C", "D", "E", "F", "J", "M", "N", "O", "P", "Q", "R", "W", "T", "1", "2", "3","10","Mau", "Bicara", "Kamu", "TerimaKasih", "U", "V","I", "X", "Y", "Z", "4","5", "6", "7","Sedih", "Maaf", "Sabar", "Sayang", "Senang", "H", "S","G" , "K", "L", "8", "9"]
max_support_per_class = 5  # Tentukan jumlah maksimal support set per kelas (ubah sesuai kebutuhan)

for cls in classes:
    cls_dir = os.path.join(app.config['SUPPORT_SET_DIR'], cls)
    if not os.path.exists(cls_dir):
        print(f"Warning: Directory for class {cls} not found at {cls_dir}")
        continue

    # Cari semua file .npy di dalam folder kelas
    npy_files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
    if not npy_files:
        print(f"Warning: No .npy files found for class {cls} in {cls_dir}")
        continue 

    # Batasi jumlah file yang diproses per kelas
    if len(npy_files) > max_support_per_class:
        npy_files = random.sample(npy_files, max_support_per_class)
        print(f"Class {cls}: Limited to {max_support_per_class} support set files: {npy_files}")
    else:
        print(f"Class {cls}: Using all {len(npy_files)} support set files: {npy_files}")

    for npy_file in npy_files:
        file_path = os.path.join(cls_dir, npy_file)
        try:
            data = np.load(file_path)
            # Pastikan data memiliki bentuk yang benar
            if data.ndim == 2 and data.shape[0] == 30 and data.shape[1] == 84:
                data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            elif data.ndim == 3 and data.shape[1] == 30 and data.shape[2] == 84:
                data = data[0]  # Ambil sampel pertama jika ada banyak
                data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            else:
                print(f"Warning: Invalid shape for file {npy_file} in class {cls}: {data.shape}. Expected (30, 84) or (N, 30, 84)")
                continue

            print(f"File {npy_file} for class {cls}: data_tensor shape = {data_tensor.shape}")
            with torch.no_grad():
                embedding = model(data_tensor).cpu().numpy()
            support_embeddings.append(embedding)
            support_labels.append(cls)
        except Exception as e:
            print(f"Error loading file {npy_file} for class {cls}: {str(e)}")
            continue

if not support_embeddings:
    raise ValueError("No valid support set data loaded. Please check .npy files.")

support_embeddings = np.concatenate(support_embeddings, axis=0)
support_labels = np.array(support_labels)
print(f"Total support embeddings: {support_embeddings.shape[0]} samples")

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    print(f"Checking file: {filename}")
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return redirect(url_for('login'))

# Konfigurasi logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email_or_username = request.form['email_or_username']
        password = request.form['password']

        users = load_users()
        user = None
        for email, data in users.items():
            if email == email_or_username or data['username'] == email_or_username:
                user = data
                break

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            session['user'] = email_or_username  # Set session for the logged-in user
#            flash('Login successful!', 'success')
            logging.info(f"User {email_or_username} logged in successfully.")
            return redirect(url_for('upload_video'))
        else:
            flash('Invalid email/username or password.', 'danger')
            logging.info(f"Failed login attempt for {email_or_username}.")

    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    print("Received request:", request.method)
    if request.method == 'POST':
        print("Files in request:", request.files)
        if 'file' not in request.files:
            print("No file part in request")
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print("File received:", file.filename)
        if file.filename == '':
            print("No selected file")
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print("File allowed, saving as:", filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print("File saved at:", filepath)
            try:
                # Proses video untuk mendapatkan keypoint
                print("Processing video keypoints...")
                keypoints = process_video_keypoints(filepath)
                if keypoints.shape != (30, 84):  # Sesuaikan dengan 30 frame
                    raise ValueError(f"Invalid keypoints shape: {keypoints.shape}. Expected (30, 84)")
                keypoints_tensor = torch.FloatTensor(keypoints).unsqueeze(0).to(device)
                print(f"Video keypoints_tensor shape = {keypoints_tensor.shape}")

                # Dapatkan embedding dari video
                print("Generating video embedding...")
                with torch.no_grad():
                    video_embedding = model(keypoints_tensor).cpu().numpy()
                print("Video embedding generated")
 
                # Bandingkan dengan support set menggunakan jarak Euclidean
                # Log jarak yang dihitung beserta labelnya
                print("Computing distances between video embedding and support set...")
                print(f"Video embedding: {video_embedding}")
                print(f"Support embeddings: {support_embeddings}")
 
                # Hitung jarak dan log dengan labelnya
                distances = euclidean_distances(video_embedding, support_embeddings)
                for idx, distance in enumerate(distances[0]):
                    print(f"Distance to {support_labels[idx]}: {distance}")
                nearest_idx = np.argmin(distances, axis=1)
                predicted_class = support_labels[nearest_idx[0]]
                print(f"Predicted class: {predicted_class}")

                # Kirim hasil ke halaman result
                print("Rendering result page")
                return render_template('result.html', prediction=predicted_class, video_file=filename)
            except Exception as e:
                print(f"Error processing video: {str(e)}")
                flash(f"Error processing video: {str(e)}")
                return redirect(request.url)
            finally:
                if os.path.exists(filepath):
                    print("Cleaning up: removing file", filepath)
                    os.remove(filepath)
        else:
            print("Invalid file format")
            flash('Invalid file format. Allowed formats: mp4, avi, mov')
            return redirect(request.url)
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        users = load_users()
        if email in users:
            flash('Email already registered. Please log in.', 'danger')
            return redirect(url_for('login'))

        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        save_user(email, username, hashed_password)
        flash('Account created successfully. Please log in.', 'success')
        logging.info(f"New account created: {email}.")
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/deteksi', methods=['GET', 'POST'])
def deteksi():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request.', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Process the video to extract keypoints
                keypoints = process_video_keypoints(filepath)
                if keypoints.shape != (30, 84):
                    raise ValueError(f"Invalid keypoints shape: {keypoints.shape}. Expected (30, 84)")

                keypoints_tensor = torch.FloatTensor(keypoints).unsqueeze(0).to(device)

                # Generate embedding for the video
                with torch.no_grad():
                    video_embedding = model(keypoints_tensor).cpu().numpy()

                # Compare with support set using Euclidean distances
                distances = euclidean_distances(video_embedding, support_embeddings)
                nearest_idx = np.argmin(distances, axis=1)
                predicted_class = support_labels[nearest_idx[0]]

 #               flash(f'Detected gesture: {predicted_class}', 'success')
                return render_template('deteksi.html', prediction=predicted_class, video_file=filename)

            except Exception as e:
                flash(f'Error processing video: {str(e)}', 'danger')
                return redirect(request.url)

            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

        else:
            flash('Invalid file format. Allowed formats: mp4, avi, mov', 'danger')
            return redirect(request.url)

    return render_template('deteksi.html')

@app.route('/belajar', methods=['GET'])
def belajar():
    return render_template('belajar.html')

@app.route('/belajar/<kategori>', methods=['GET'])
def belajar_kategori(kategori):
    # Tentukan folder berdasarkan kategori
    kategori_map = {
        'angka': [str(i) + '.gif' for i in range(1, 11)],
        'huruf': [chr(i) + '.gif' for i in range(65, 91)],  # A-Z
        'kata': ['Aku.gif', 'Cinta.gif', 'Kenapa.gif', 'Tidak.gif', 'Kecewa.gif']
    }

    if kategori not in kategori_map:
        return {'error': 'Kategori tidak ditemukan.'}, 404

    folder_path = 'test_case'
    gif_files = []

    # Ambil file gif yang sesuai dengan pola kategori
    if os.path.exists(folder_path):
        all_files = os.listdir(folder_path)
        gif_files = [f for f in all_files if f in kategori_map[kategori]]

    return {'video_files': gif_files}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)