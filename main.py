from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from multiprocessing import Process
import time
import cv2

from OOP2_11.facedetect import detect_faces

app = Flask(__name__)

# アップロードされた画像を保存するディレクトリ
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER



# 許可されたファイルの拡張子をチェックする関数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ファイルをアップロードして処理を行う関数
def process_image(file_path):
    base_filename = os.path.basename(file_path)
    unique_suffix = str(int(time.time()))
    # 画像処理のサンプル（ここでは単純にグレースケール化）
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(file_path))
    cv2.imwrite(processed_path, gray_image)

    face_detected_image_path = os.path.join(app.config['PROCESSED_FOLDER'],
                                            "face_" + unique_suffix + "_" + base_filename)
    detect_faces(file_path, face_detected_image_path)

# ファイルアップロード用のエンドポイント
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

# 処理された画像一覧のエンドポイント
@app.route('/processed')
def processed_images():
    processed_files = os.listdir(app.config['PROCESSED_FOLDER'])
    return render_template('processed.html', processed_files=processed_files)

if __name__ == '__main__':
    app.run(debug=True)