
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

def create_plot():


    N = 40
    x = np.linspace(0, 1, N)
    y = np.random.randn(N)
    df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe


    data = [
        go.Bar(
            x=df['x'], # assign x as the dataframe column 'x'
            y=df['y']
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

## to get current year
@app.context_processor
def inject_now():
    return {'now': datetime.utcnow()}

@app.route("/")
def home():
    bar = create_plot()
    return render_template("index.html", graphJSON=bar)
        

@app.route("/UnderstandML")
def UnderstandML():
    return render_template("UnderstandML.html")

@app.route("/About")
def About():
    return render_template("About.html")


if __name__ == "__main__":
    app.run(debug=True)