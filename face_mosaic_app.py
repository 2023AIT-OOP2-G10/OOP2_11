import cv2
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from multiprocessing import Process

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Haar Cascade分類器の読み込み
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(file_path):
    # 画像読み込み
    image = cv2.imread(file_path)
    
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 顔検出
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # 顔にモザイク処理
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.GaussianBlur(face, (99, 99), 30)  # ガウシアンブラーをかけてモザイク化
        image[y:y+h, x:x+w] = face
    
    # 保存
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(file_path))
    cv2.imwrite(processed_path, image)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # 別プロセスで画像処理を行う
            process = Process(target=process_image, args=(file_path,))
            process.start()
            
            return redirect(url_for('upload_file'))
    return render_template('upload.html')

@app.route('/processed')
def processed_images():
    processed_files = os.listdir(app.config['PROCESSED_FOLDER'])
    return render_template('processed.html', processed_files=processed_files)

if __name__ == '__main__':
    app.run(debug=True)
