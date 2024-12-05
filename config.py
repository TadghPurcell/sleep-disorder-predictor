import os
from datetime import datetime

class Config:
    # SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    HADOOP_STREAMING_JAR = '/usr/local/hadoop-3.3.6/share/hadoop/tools/lib/hadoop-streaming-3.3.6.jar'
    HDFS_INPUT_PATH = '/healthcare_data/sleep_health_dataset.csv'
    HDFS_OUTPUT_PATH = f'/output/sleep_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

