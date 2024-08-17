from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
#from mlProject.pipeline.prediction import PredictionPipeline


app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 






if __name__== "__main__":
    app.run(host="0.0.0.0",port = 8080)


