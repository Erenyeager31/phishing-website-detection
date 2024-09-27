import streamlit as st
import pandas as pd
import os
import pickle
import tldextract
from urllib.parse import urlparse
import socket
import dns.resolver
import time

# Load the trained model and scaler from pickle files
model_filename = 'phishing_website_detection_model.pkl'
scaler_filename = 'scaler.pkl'

# Construct the paths dynamically
model_filename = os.path.join(os.path.dirname(__file__), '..', model_filename)
scaler_filename = os.path.join(os.path.dirname(__file__), '..', scaler_filename)

# Convert paths to absolute paths
model_filename = os.path.abspath(model_filename)
scaler_filename = os.path.abspath(scaler_filename)

# Load the model
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open(scaler_filename, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def extract_features(url):
    start_time = time.time()
    
    extracted = tldextract.extract(url)
    parsed_url = urlparse(url)
    
    # Calculate features
    qty_dot_url = url.count('.')
    qty_hyphen_url = url.count('-')
    qty_slash_url = url.count('/')
    qty_dot_domain = (extracted.domain + extracted.suffix).count('.')
    qty_hyphen_domain = (extracted.domain + extracted.suffix).count('-')
    qty_dot_directory = parsed_url.path.count('.')
    qty_hyphen_directory = parsed_url.path.count('-')
    length_url = len(url)
    domain_length = len(extracted.domain + extracted.suffix)
    directory_length = len(parsed_url.path)
    file_length = len(parsed_url.path.split('/')[-1]) if parsed_url.path else 0

    # Resolve IP addresses
    try:
        ip_addresses = socket.gethostbyname_ex(extracted.registered_domain)[2]
        qty_ip_resolved = len(ip_addresses)
    except socket.gaierror:
        qty_ip_resolved = 0

    # Resolve nameservers
    try:
        nameservers = dns.resolver.resolve(extracted.registered_domain, 'NS')
        qty_nameservers = len(nameservers)
    except dns.exception.DNSException:
        qty_nameservers = 0

    # Calculate response time
    time_response = time.time() - start_time

    return pd.DataFrame({
        'qty_dot_url': [qty_dot_url],
        'qty_hyphen_url': [qty_hyphen_url],
        'qty_slash_url': [qty_slash_url],
        'qty_dot_domain': [qty_dot_domain],
        'qty_hyphen_domain': [qty_hyphen_domain],
        'qty_dot_directory': [qty_dot_directory],
        'qty_hyphen_directory': [qty_hyphen_directory],
        'length_url': [length_url],
        'domain_length': [domain_length],
        'directory_length': [directory_length],
        'file_length': [file_length],
        'time_response': [time_response],
        'qty_ip_resolved': [qty_ip_resolved],
        'qty_nameservers': [qty_nameservers]
    })

# Streamlit UI components
st.title("Phishing Website Detection")
st.write("Enter the URL to check if it's phishing or not:")

# Input field for URL
url_input = st.text_input("URL")

# Button to make prediction
if st.button("Check URL"):
    if url_input:
        # Extract features from the URL
        input_data = extract_features(url_input)

        st.write("Extracted Features:")
        st.write(input_data)
        
        # Preprocess input using the loaded scaler
        input_scaled = scaler.transform(input_data)
        
        st.write("Scaled Features:")
        st.write(pd.DataFrame(input_scaled, columns=input_data.columns))
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Display the result
        if prediction[0] == 1:
            st.error("⚠️ Warning: This URL is likely a phishing website!")
        else:
            st.success("✅ This URL appears to be safe.")
        
        # Display prediction probability
        prediction_proba = model.predict_proba(input_scaled)
        st.write(f"Probability of being a phishing website: {prediction_proba[0][1]:.2%}")
        
    else:
        st.error("Please enter a valid URL.")

# https://github.com/gangeshbaskerr/Phishing-Website-Detection/blob/main/Phishing%20Website%20Detection_Models%20%26%20Training.ipynb