# IoT-Device-Fingerprinting-and-Authentication-Using-Behavioral-Biometrics
In an IoT network, devices often exhibit unique behavior based on how they communicate with the network (e.g., packet transmission rate, message frequency, and data request patterns). This project uses these unique behavioral characteristics to create device fingerprints for authentication,
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import time
import random

# Simulate IoT device network traffic behavior
def generate_device_traffic():
    """
    Simulate normal and spoofed IoT device network behavior.
    """
    normal_traffic = np.random.normal(loc=100, scale=15, size=(100, 3))  # Normal traffic behavior
    spoofed_traffic = np.random.normal(loc=500, scale=100, size=(10, 3))  # Spoofed or impersonated behavior
    
    data = np.vstack([normal_traffic, spoofed_traffic])
    return pd.DataFrame(data, columns=["packet_size", "request_rate", "connection_duration"])

# Train an anomaly detection model using One-Class SVM for device fingerprinting
def train_device_fingerprint_model(data):
    """
    Train a One-Class SVM model for IoT device fingerprinting based on network behavior.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    model = OneClassSVM(nu=0.1, kernel="rbf", gamma="auto")
    model.fit(data_scaled)
    
    return model, scaler

# Monitor and authenticate IoT devices based on network behavior
def monitor_device_traffic(model, scaler):
    """
    Continuously monitor IoT device traffic and authenticate based on behavior.
    """
    while True:
        # Simulate new device network traffic
        new_device_traffic = np.random.normal(loc=100, scale=15, size=(1, 3))  # Normal traffic
        
        # Occasionally simulate spoofed traffic
        if random.random() > 0.95:
            new_device_traffic = np.random.normal(loc=500, scale=100, size=(1, 3))  # Spoofed traffic
        
        # Standardize and classify the new device traffic
        new_device_scaled = scaler.transform(new_device_traffic)
        prediction = model.predict(new_device_scaled)
        
        if prediction == -1:
            print(f"[ALERT] Impersonation or spoofing detected! Suspicious traffic: {new_device_traffic}")
        else:
            print(f"[INFO] Authentic device traffic: {new_device_traffic}")
        
        time.sleep(2)  # Simulate real-time monitoring delay

if __name__ == "__main__":
    # Step 1: Generate IoT device network traffic
    print("Generating IoT device network traffic data...")
    device_traffic_data = generate_device_traffic()

    # Step 2: Train the device fingerprinting model
    print("Training device fingerprinting model...")
    model, scaler = train_device_fingerprint_model(device_traffic_data)

    # Step 3: Monitor device traffic in real-time for authentication
    print("Monitoring IoT device traffic for authentication...")
    monitor_device_traffic(model, scaler)
