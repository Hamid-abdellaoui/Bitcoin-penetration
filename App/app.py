
from flask import current_app, send_file, send_from_directory
import sys 
import os
import plotly
import plotly.express as px
import json
import plotly.graph_objs as go
import pandas as pd
import numpy as np

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
    min_month = "2012-11"
    max_month = "2022-5"
    months = pd.period_range(min_month, max_month, freq='M')
    value = months.to_timestamp(how='end').strftime('%Y-%m')


    return render_template("index.html", months=value)
        

@app.route("/topics")
def topics():
    return render_template("topics.html")

@app.route("/About")
def About():
    return render_template("About.html")


if __name__ == "__main__":
    app.run(debug=True)