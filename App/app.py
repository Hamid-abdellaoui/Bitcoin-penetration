
from flask import current_app, send_file, send_from_directory
import sys 
import os

from flask import Flask, render_template,request 
from datetime import datetime
import pandas as pd
from werkzeug.utils import secure_filename 
app = Flask(__name__)


import seaborn as se
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np 


k=0



## to get current year
@app.context_processor
def inject_now():
    return {'now': datetime.utcnow()}

@app.route("/")
def home():
    return render_template("index.html")
        

@app.route("/UnderstandML")
def UnderstandML():
    return render_template("UnderstandML.html")

@app.route("/About")
def About():
    return render_template("About.html")


if __name__ == "__main__":
    app.run(debug=True)