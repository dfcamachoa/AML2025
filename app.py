import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Industrial Boiler Anomaly Detection", layout="wide")

# Title and description
st.title("Coal-Fired Industrial Boiler Anomaly Detection")
st.markdown("""
This application analyzes time-series data from a coal-fired industrial boiler in a chemical plant.
It uses an autoencoder model to detect anomalies in the sensor readings.
Approximately 8.6% of the data represents abnormal operating conditions.
""")

# Function to load data
@st.cache_data
def load_data(file):
    try:
        # Custom date parser to handle the format in the dataset
        date_parser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        
        # Read the CSV file
        df = pd.read_csv(file, parse_dates=['date'], date_parser=date_parser)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# File uploader
uploaded_file = st.file_uploader("Upload your boiler time-series dataset (CSV)", type="csv")

# Function to preprocess data for the autoencoder
def preprocess_data(df):
    # Drop any rows with missing values
    df_clean = df.dropna()
    
    # Extract features (all columns except date)
    features = df_clean.columns.drop('date')
    X = df_clean[features].values
    
    # Normalize data between 0 and 1
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, features, scaler, df_clean

# Function to build autoencoder model
def build_autoencoder(input_dim, encoding_dim=10):
    # Input layer
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoder = Dense(int(input_dim*0.75), activation='relu')(input_layer)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(int(input_dim*0.5), activation='relu')(encoder)
    encoder = Dropout(0.2)(encoder)
    encoder = Dense(encoding_dim, activation='relu')(encoder)
    
    # Decoder
    decoder = Dense(int(input_dim*0.5), activation='relu')(encoder)
    decoder = Dropout(0.2)(decoder)
    decoder = Dense(int(input_dim*0.75), activation='relu')(decoder)
    decoder = Dropout(0.2)(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    
    # Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    
    # Encoder model
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    
    # Compile model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    return autoencoder, encoder_model

# Main application logic
if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)
    
    if df is not None:
        # Display basic information
        st.subheader("Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"Number of records: {len(df)}")
        with col2:
            st.info(f"Time period: {df['date'].min()} to {df['date'].max()}")
        with col3:
            st.info(f"Number of sensors: {len(df.columns) - 1}")  # -1 for date column
        
        # Preprocess data
        X_scaled, features, scaler, df_clean = preprocess_data(df)
        
        # Display tabs for different parts of the analysis
        tab1, tab2, tab3, tab4 = st.tabs(["Data Exploration", "Autoencoder Training", "Anomaly Detection", "Feature Analysis"])
        
        with tab1:
            st.header("Data Exploration")
            
            # Sample data
            st.subheader("Sample Data")
            st.dataframe(df.head())
            
            # Time series plots
            st.subheader("Time Series Visualization")
            
            # Select features to visualize
            selected_features = st.multiselect(
                "Select sensors to visualize",
                options=features,
                default=features[:3]  # Default to first 3 features
            )
            
            if selected_features:
                # Create time series plot with Plotly
                fig = go.Figure()
                for feature in selected_features:
                    fig.add_trace(go.Scatter(
                        x=df_clean['date'],
                        y=df_clean[feature],
                        mode='lines',
                        name=feature
                    ))
                
                fig.update_layout(
                    title="Sensor Readings Over Time",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Correlation Analysis")
            
            # Calculate correlation matrix
            corr = df_clean[features].corr()
            
            # Plot heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)
            
            plt.title("Correlation Between Sensors")
            st.pyplot(fig)
            
            # Basic statistics
            st.subheader("Descriptive Statistics")
            st.dataframe(df_clean[features].describe())
        
        with tab2:
            st.header("Autoencoder Model Training")
            
            # Model parameters
            col1, col2 = st.columns(2)
            with col1:
                encoding_dim = st.slider("Encoding dimension", min_value=2, max_value=20, value=10)
                batch_size = st.slider("Batch size", min_value=16, max_value=256, value=64)
            with col2:
                epochs = st.slider("Training epochs", min_value=5, max_value=100, value=20)
                validation_split = st.slider("Validation split", min_value=0.1, max_value=0.3, value=0.2)
            
            # Train model button
            if st.button("Train Autoencoder Model"):
                with st.spinner("Training model..."):
                    # Split data into train and test sets
                    X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
                    
                    # Build and train model
                    input_dim = X_train.shape[1]
                    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
                    
                    # Display model summary
                    st.text("Autoencoder Model Architecture:")
                    autoencoder.summary(print_fn=lambda x: st.text(x))
                    
                    # Train model
                    history = autoencoder.fit(
                        X_train, X_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        verbose=0
                    )
                    
                    # Plot training history
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(history.history['loss'], label='Training Loss')
                    ax.plot(history.history['val_loss'], label='Validation Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title('Autoencoder Training and Validation Loss')
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Save model to session state
                    st.session_state['autoencoder'] = autoencoder
                    st.session_state['encoder'] = encoder
                    st.session_state['X_scaled'] = X_scaled
                    st.session_state['X_test'] = X_test
                    
                    st.success("Model training complete!")
        
        with tab3:
            st.header("Anomaly Detection")
            
            if 'autoencoder' not in st.session_state:
                st.warning("Please train the model in the 'Autoencoder Training' tab first.")
            else:
                autoencoder = st.session_state['autoencoder']
                X_scaled = st.session_state['X_scaled']
                
                # Calculate reconstruction error
                pred = autoencoder.predict(X_scaled)
                mse = np.mean(np.power(X_scaled - pred, 2), axis=1)
                
                # Plot reconstruction error distribution
                fig = px.histogram(
                    mse, 
                    nbins=100, 
                    title="Distribution of Reconstruction Error",
                    labels={'value': 'Reconstruction Error', 'count': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Set threshold for anomaly detection
                threshold = st.slider(
                    "Select threshold for anomaly detection", 
                    min_value=float(np.min(mse)), 
                    max_value=float(np.max(mse)), 
                    value=float(np.percentile(mse, 91.4)),  # Default to 91.4 percentile (since 8.6% are anomalies)
                    format="%.5f"
                )
                
                # Identify anomalies
                anomalies = mse > threshold
                
                # Add reconstruction error and anomaly flag to dataframe
                df_results = df_clean.copy()
                df_results['reconstruction_error'] = mse
                df_results['is_anomaly'] = anomalies
                
                # Display results
                st.subheader("Anomaly Detection Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"Total data points: {len(df_results)}")
                with col2:
                    st.info(f"Normal points: {len(df_results) - sum(anomalies)}")
                with col3:
                    st.info(f"Anomalies detected: {sum(anomalies)} ({sum(anomalies)/len(df_results):.2%})")
                
                # Plot time series with anomalies highlighted
                st.subheader("Time Series with Anomalies")
                
                # Select feature to visualize
                selected_feature = st.selectbox(
                    "Select sensor to visualize",
                    options=features
                )
                
                # Create plot
                fig = go.Figure()
                
                # Add normal points
                fig.add_trace(go.Scatter(
                    x=df_results[~anomalies]['date'],
                    y=df_results[~anomalies][selected_feature],
                    mode='markers',
                    name='Normal',
                    marker=dict(color='blue', size=4, opacity=0.6)
                ))
                
                # Add anomalies
                fig.add_trace(go.Scatter(
                    x=df_results[anomalies]['date'],
                    y=df_results[anomalies][selected_feature],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='red', size=8)
                ))
                
                fig.update_layout(
                    title=f"Anomaly Detection for {selected_feature}",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display anomalies
                if st.checkbox("Show anomalous data points"):
                    st.dataframe(df_results[anomalies].sort_values('reconstruction_error', ascending=False))
        
        with tab4:
            st.header("Feature Analysis")
            
            if 'autoencoder' not in st.session_state:
                st.warning("Please train the model in the 'Autoencoder Training' tab first.")
            else:
                autoencoder = st.session_state['autoencoder']
                encoder = st.session_state['encoder']
                X_scaled = st.session_state['X_scaled']
                
                # Calculate feature reconstruction errors
                pred = autoencoder.predict(X_scaled)
                feature_errors = np.mean(np.power(X_scaled - pred, 2), axis=0)
                
                # Plot feature reconstruction errors
                feature_error_df = pd.DataFrame({
                    'Feature': features,
                    'Reconstruction Error': feature_errors
                }).sort_values('Reconstruction Error', ascending=False)
                
                fig = px.bar(
                    feature_error_df,
                    x='Feature',
                    y='Reconstruction Error',
                    title="Average Reconstruction Error by Feature",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Latent space visualization
                st.subheader("Latent Space Visualization")
                
                # Encode data to latent space
                encoded_data = encoder.predict(X_scaled)
                
                # If we have trained the model with more than 2 dimensions in the latent space,
                # allow the user to select which dimensions to visualize
                if encoding_dim > 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        dim1 = st.selectbox("Select first dimension", options=range(encoding_dim), index=0)
                    with col2:
                        dim2 = st.selectbox("Select second dimension", options=range(encoding_dim), index=1)
                    
                    # Create scatter plot of selected dimensions
                    fig = px.scatter(
                        x=encoded_data[:, dim1],
                        y=encoded_data[:, dim2],
                        title=f"Latent Space Visualization (Dimensions {dim1} and {dim2})",
                        labels={'x': f'Dimension {dim1}', 'y': f'Dimension {dim2}'}
                    )
                else:
                    # If latent space is 2D, just plot those dimensions
                    fig = px.scatter(
                        x=encoded_data[:, 0],
                        y=encoded_data[:, 1],
                        title="2D Latent Space Visualization",
                        labels={'x': 'Dimension 0', 'y': 'Dimension 1'}
                    )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Instructions when no data is uploaded
else:
    st.info("Please upload a CSV file containing the boiler time-series data to get started.")
    
    # Example of the expected data format
    st.subheader("Expected Data Format Example:")
    example_data = pd.DataFrame({
        'date': ['3/27/2022 14:28', '3/27/2022 14:28', '3/27/2022 14:29'],
        'PT_8313A.AV_0#': [-142.28, -142.79, -135.23],
        'PT_8313B.AV_0#': [-179.46, -187.01, -195.32],
        'TE_8319A.AV_0#': [352.50, 352.50, 352.50]
    })
    st.dataframe(example_data)
    
    st.markdown("""
    ### Key Features:
    
    1. **Data Exploration**: Visualize time series data and analyze correlations between sensors
    2. **Autoencoder Training**: Train a neural network to learn normal operating patterns
    3. **Anomaly Detection**: Identify abnormal operating conditions based on reconstruction error
    4. **Feature Analysis**: Understand which sensors contribute most to anomalies
    
    Upload your data to begin!
    """)

# Add footer
st.markdown("---")
st.markdown("Coal-Fired Industrial Boiler Anomaly Detection | Developed with Streamlit and TensorFlow")