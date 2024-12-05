from flask import Blueprint, render_template, jsonify, request
from app.mapreduce.sleep_predictor import SleepDisorderPredictor
import subprocess
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
import os
import plotly
from config import Config
import plotly.express as px
from pyarrow import fs
from dotenv import load_dotenv

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s'
)

HDFS_HOST = os.getenv("HDFS_HOST", "127.0.0.1")
HDFS_PORT = int(os.getenv("HDFS_PORT", 8020))
HDFS_USER = os.getenv("HDFS_USER", "hadoop")  

app = Blueprint('main', __name__)

@app.route('/')
def index():
    # Read and process the CSV data for initial visualization
    df = pd.read_csv('data/sleep_health_dataset.csv')
    
    # Create some basic visualizations
    # Pie chart of sleep disorders
    fig1 = px.pie(values=df['Sleep Disorder'].value_counts().values, 
                  names=df['Sleep Disorder'].value_counts().index, 
                  title='Distribution of Sleep Disorders')
    
    # Bar chart of average sleep duration
    avg_sleep = df.groupby('Sleep Disorder')['Sleep Duration'].mean()
    fig2 = px.bar(x=avg_sleep.index, 
                  y=avg_sleep.values,
                  title='Average Sleep Duration by Disorder Type')

    # Generate HTML directly from the figures
    plot1_html = fig1.to_html(full_html=False, include_plotlyjs=False)
    plot2_html = fig2.to_html(full_html=False, include_plotlyjs=False)
    
    return render_template('index.html', 
                         plot1=plot1_html, 
                         plot2=plot2_html)

@app.route('/run-mapreduce', methods=['POST'])
def run_mapreduce():
    try:
        # Run MapReduce job
        cmd = [
            'hadoop', 'jar', Config.HADOOP_STREAMING_JAR,
            '-files', 'app/mapreduce/sleep_predictor.py',
            '-mapper', 'sleep_predictor.py --mapper',
            '-reducer', 'sleep_predictor.py --reducer',
            '-input', Config.HDFS_INPUT_PATH,
            '-output', Config.HDFS_OUTPUT_PATH
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Read results from HDFS
        cat_process = subprocess.Popen(
            ['hadoop', 'fs', '-cat', f'{Config.HDFS_OUTPUT_PATH}/part-*'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, err = cat_process.communicate()
        
        results = json.loads(output.decode('utf-8'))
        return jsonify(success=True, results=results)
    
    except Exception as e:
        return jsonify(success=False, error=str(e))
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        logging.info(f"Received data: {data}")
        
        # Create feature vector
        features = np.array([[
            float(data['sleep_duration']),
            float(data['quality_of_sleep']),
            float(data['physical_activity']),
            float(data['stress_level']),
            float(data['heart_rate']),
            float(data['daily_steps'])
        ]])
        
        # Load training data for scaling
        df = pd.read_csv('data/sleep_health_dataset.csv')
        X_train = df[['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
                      'Stress Level', 'Heart Rate', 'Daily Steps']].values
        
        # Scale features
        scaler = StandardScaler()
        scaler.fit(X_train)
        features_scaled = scaler.transform(features)
        
        # Calculate risk score (using the same weights as in MapReduce)
        risk_score = (
            0.3 * features_scaled[0, 0] +  # Sleep Duration
            0.2 * features_scaled[0, 1] +  # Quality of Sleep
            0.1 * features_scaled[0, 2] +  # Physical Activity
            0.2 * features_scaled[0, 3] +  # Stress Level
            0.1 * features_scaled[0, 4] +  # Heart Rate
            0.1 * features_scaled[0, 5]    # Daily Steps
        )
        
        # Make prediction
        if risk_score > 0.5:
            prediction = "High risk of sleep disorder"
            if data['stress_level'] > '7' and float(data['sleep_duration']) < 6:
                disorder_type = "Likely Insomnia"
            else:
                disorder_type = "Likely Sleep Apnea"
            confidence = min(abs(risk_score) * 100, 100)
        else:
            prediction = "Low risk of sleep disorder"
            disorder_type = "No Sleep Disorder"
            confidence = min((1 - risk_score) * 100, 100)
        
        return jsonify({
            'prediction': f"{prediction} - {disorder_type}",
            'confidence': f"{confidence:.1f}"
        })
        
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/hdfs_test')
def hdfs_test():
    try:
        hdfs = fs.HadoopFileSystem(
            host=HDFS_HOST,
            port=HDFS_PORT,
            user=HDFS_USER,
            kerb_ticket=KERB_TICKET
        )
        
        # List files in the root directory
        files = hdfs.get_file_info(fs.FileSelector('/'))
        file_list = [file.path for file in files]
        
        return jsonify({"status": "success", "files": file_list})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500