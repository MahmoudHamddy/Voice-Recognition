from email import message
from flask import Flask, redirect, url_for, request,render_template,flash,session
import numpy as np
from sympy import Id



app = Flask(__name__,template_folder="templates")

@app.route('/')
def hello_name():
   return render_template('index.html')

if __name__ == '__main__':
   app.run(debug=True)   