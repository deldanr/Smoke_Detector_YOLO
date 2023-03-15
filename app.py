#############################################
# SMOKE DETECTOR V2.0 "SMOKY"
#
# Author: Daniel Eldan R.
# Date  : 12-2022
# Mail  : deldanr@gmail.com
# Name  : Main
# Desc  : Load the app - Run Flask Server and run background function to scrap images
############################################

#
# LOAD LIBRARIES
#

from flask import Flask, flash, request, redirect, url_for, render_template
import os
import shutil
import subprocess
import time

from multiprocessing import Process
from werkzeug.utils import secure_filename

from tinydb import TinyDB, Query

db = TinyDB("static/data.json")
Todo = Query()


#
# Local libraries
#
import Scrapper as sc

app = Flask(__name__)

#
# GLOBAL VARIABLES
#
UPLOAD_FOLDER = 'static/test/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = "secret key"

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

#
# Check if uploaded image is in allowed extensions
#
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#
# Render index.html
#
@app.route('/')
def home():
    Detectadas = db.all()[0]["Detectadas"]
    Procesadas = db.all()[0]["Procesadas"]
    return render_template('index.html', Proce=Procesadas, Detec=Detectadas)

#
# Render camaras.html
#
@app.route('/camaras')
def camaras():

    return render_template("camaras.html")

#
# Function to run inference on an upload image from index.html
#
@app.route('/', methods=['POST'])
def upload_image():
    
    # Check if the image is marked as false positive and save it to its folder
    if request.form.get('fpositivo'):
        ruta = UPLOAD_FOLDER+request.form.get('fpositivo')
        shutil.copy2(ruta, "static/fpositivo/")
        flash("Falso positivo informado correctamente")
        return redirect(request.url)

    # Check if the image is marked as false negative and save it to its folder
    if request.form.get('fnegativo'):
        ruta = UPLOAD_FOLDER+request.form.get('fnegativo')
        shutil.copy2(ruta, "static/fnegativo/")
        flash("Falso negativo informado correctamente")
        return redirect(request.url)
    
    # Check if there is a file uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    # If we got an image uploaded with the allowed extensions:
    if file and allowed_file(file.filename):
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)

        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        preruta = 'C:/Users/delda/OneDrive/Escritorio/Proyectos/Smoke_Detector2.0/static/test/uploads/'
        ruta = os.path.abspath(preruta + filename)
        
        # Run inference on the uploaded image

        subprocess.call(["python", "detect.py", "--name", "uploads", "--exist-ok", "--source", ruta])
        time.sleep(5)
        return render_template('index.html', filename=filename)
       
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
   

#
# Shut down server
#
@app.route('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'
    
    

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='test/uploads/' + filename), code=301)


#
# Function to stop server
#
def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()  
#
# Main function to run the Flask server
#
def run_app():
    print("******************************************************************")
    print("******************** INICIANDO SMOKY DETECTOR ********************")
    print("******************************************************************")
    app.run(
        host = "0.0.0.0",
        port = 5000,
        debug = False,
        threaded=True # Important for backgrounding
    )

#
# To run the Flask Server and automatically update the images to analyse
#
def parallelize_functions(*functions):
    processes = []
    for function in functions:
        p = Process(target=function)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

#
# Run the app
#
if __name__ == '__main__':
    parallelize_functions(sc.actualiza_imagenes, run_app)
    

