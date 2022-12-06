from email import message
from flask import Flask, redirect, url_for, request,render_template,flash,session
import numpy as np
from sympy import Id
import pickle




app = Flask(__name__,template_folder="templates")
#model = pickle.load(open("DSP_Task3_TeamNo-main.rar", "rb"))

@app.route('/')
def hello_name():
   return render_template('index.html')

# @app.route('/upload')
# def upload_file():
#    return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      return 'file uploaded successfully'

if __name__ == '__main__':
   app.run(debug=True)   
