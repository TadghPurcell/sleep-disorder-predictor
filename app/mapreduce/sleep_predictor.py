from mrjob.job import MRJob
from mrjob.step import MRStep
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler

class SleepDisorderPredictor(MRJob):
    
    def mapper_init(self):
        # Initialize feature names
        self.features = ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
                        'Stress Level', 'Heart Rate', 'Daily Steps']
        self.scaler = StandardScaler()
    
    def mapper(self, _, line):
        # Skip header
        if 'Person ID' in line:
            return
        
        # Parse CSV line
        row = next(csv.reader([line]))
        try:
            # Extract features
            features = [float(row[4]),  # Sleep Duration
                       float(row[5]),  # Quality of Sleep
                       float(row[6]),  # Physical Activity Level
                       float(row[7]),  # Stress Level
                       float(row[11]), # Heart Rate
                       float(row[12])] # Daily Steps
            
            # Extract label
            sleep_disorder = row[13]  # Sleep Disorder status
            
            # Emit features and label for each record
            yield "data", (features, sleep_disorder)
            
        except (ValueError, IndexError):
            self.increment_counter('Map', 'Bad Records', 1)
    
    def reducer(self, key, values):
        # Collect all data
        features_list = []
        labels = []
        
        for features, label in values:
            features_list.append(features)
            labels.append(1 if label != "No Sleep Disorder" else 0)
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Simple threshold-based prediction
        predictions = []
        for sample in X_scaled:
            # Create risk score based on weighted features
            risk_score = (
                0.3 * sample[0] +  # Sleep Duration
                0.2 * sample[1] +  # Quality of Sleep
                0.1 * sample[2] +  # Physical Activity
                0.2 * sample[3] +  # Stress Level
                0.1 * sample[4] +  # Heart Rate
                0.1 * sample[5]    # Daily Steps
            )
            
            # Predict disorder if risk score exceeds threshold
            prediction = 1 if risk_score > 0.5 else 0
            predictions.append(prediction)
        
        # Calculate metrics
        accuracy = np.mean(np.array(predictions) == y)
        tp = np.sum((np.array(predictions) == 1) & (y == 1))
        fp = np.sum((np.array(predictions) == 1) & (y == 0))
        tn = np.sum((np.array(predictions) == 0) & (y == 0))
        fn = np.sum((np.array(predictions) == 0) & (y == 1))
        
        # Emit results
        yield "metrics", {
            "accuracy": float(accuracy),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn)
        }

    def steps(self):
        return [
            MRStep(mapper_init=self.mapper_init,
                  mapper=self.mapper,
                  reducer=self.reducer)
        ]

if __name__ == '__main__':
    SleepDisorderPredictor.run()