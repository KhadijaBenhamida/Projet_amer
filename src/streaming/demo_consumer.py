"""
Simple Kafka Consumer for Climate Predictions

This consumer reads climate data from Kafka, makes predictions using a trained model,
and logs the results. Simplified version compatible with Windows.
"""

import json
import logging
import pickle
import time
from pathlib import Path
import sys

import pandas as pd
import numpy as np
from kafka import KafkaConsumer
from kafka.errors import KafkaError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClimatePredictor:
    """Simple climate prediction consumer."""
    
    def __init__(self, bootstrap_servers='localhost:9092', topic='climate-data', 
                 model_path='models/baselines/linear_model_sklearn.pkl',
                 scaler_path='data/processed/splits/scaler_new.pkl',
                 imputer_path='data/processed/splits/imputer_new.pkl'):
        """
        Initialize the consumer and load model artifacts.
        
        Args:
            bootstrap_servers (str): Kafka broker address
            topic (str): Kafka topic to consume from
            model_path (str): Path to trained model
            scaler_path (str): Path to fitted scaler
            imputer_path (str): Path to fitted imputer
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer = None
        
        # Load model and preprocessing objects
        self._load_artifacts(model_path, scaler_path, imputer_path)
        
        # Initialize consumer
        self._connect()
        
        # Statistics
        self.messages_processed = 0
        self.predictions_made = 0
        self.errors = 0
        self.start_time = time.time()
        
    def _load_artifacts(self, model_path, scaler_path, imputer_path):
        """Load model and preprocessing artifacts."""
        try:
            # Load model
            logger.info(f"Loading model from {model_path}")
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✅ Model loaded: {type(self.model).__name__}")
            
            # Load scaler
            logger.info(f"Loading scaler from {scaler_path}")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Get expected features from scaler
            self.expected_features = list(self.scaler.feature_names_in_)
            logger.info(f"✅ Scaler loaded - expects {len(self.expected_features)} features")
            
            # Load imputer
            logger.info(f"Loading imputer from {imputer_path}")
            with open(imputer_path, 'rb') as f:
                self.imputer = pickle.load(f)
            logger.info(f"✅ Imputer loaded")
            
        except Exception as e:
            logger.error(f"❌ Error loading artifacts: {e}")
            raise
            
    def _connect(self):
        """Connect to Kafka broker."""
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                group_id='climate-predictor-group',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                consumer_timeout_ms=10000  # 10 second timeout
            )
            logger.info(f"✅ Connected to Kafka topic '{self.topic}'")
        except KafkaError as e:
            logger.error(f"❌ Failed to connect to Kafka: {e}")
            raise
            
    def _preprocess_features(self, record):
        """
        Extract and preprocess features from Kafka message.
        
        Args:
            record (dict): Raw message from Kafka
            
        Returns:
            np.ndarray: Preprocessed features ready for prediction
        """
        # Remove metadata fields
        features = {k: v for k, v in record.items() 
                   if not k.startswith('_') and k != 'temperature'}
        
        # Extract only the expected features in the correct order
        feature_values = [features.get(f, np.nan) for f in self.expected_features]
        
        # Convert to DataFrame for preprocessing
        df = pd.DataFrame([feature_values], columns=self.expected_features)
        
        # Handle missing values
        df_imputed = pd.DataFrame(
            self.imputer.transform(df),
            columns=self.expected_features
        )
        
        # Scale features
        df_scaled = pd.DataFrame(
            self.scaler.transform(df_imputed),
            columns=self.expected_features
        )
        
        return df_scaled.values
        
    def predict(self, features):
        """
        Make prediction using the loaded model.
        
        Args:
            features (np.ndarray): Preprocessed features
            
        Returns:
            float: Predicted temperature
        """
        prediction = self.model.predict(features)[0]
        return prediction
        
    def process_message(self, message):
        """
        Process a single Kafka message.
        
        Args:
            message: Kafka message object
        """
        try:
            record = message.value
            
            # Preprocess features
            features = self._preprocess_features(record)
            
            # Make prediction
            prediction = self.predict(features)
            
            # Get actual value if available
            actual = record.get('temperature', None)
            
            # Log result
            self.predictions_made += 1
            if self.predictions_made <= 10 or self.predictions_made % 100 == 0:
                if actual is not None and not pd.isna(actual):
                    error = abs(prediction - actual)
                    logger.info(f"  🎯 Prediction #{self.predictions_made}: "
                              f"Predicted={prediction:.2f}°C, Actual={actual:.2f}°C, "
                              f"Error={error:.2f}°C")
                else:
                    logger.info(f"  🎯 Prediction #{self.predictions_made}: "
                              f"Predicted={prediction:.2f}°C (no actual value)")
            
            self.messages_processed += 1
            
        except Exception as e:
            self.errors += 1
            logger.error(f"  ❌ Error processing message: {e}")
            
    def run(self, max_messages=None):
        """
        Start consuming messages and making predictions.
        
        Args:
            max_messages (int): Maximum number of messages to process (None = unlimited)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 STARTING CLIMATE PREDICTOR")
        logger.info(f"{'='*60}")
        logger.info(f"  Topic: {self.topic}")
        logger.info(f"  Max messages: {max_messages or 'Unlimited'}")
        logger.info(f"  Expected features: {len(self.expected_features)}")
        logger.info(f"{'='*60}\n")
        
        try:
            for message in self.consumer:
                self.process_message(message)
                
                # Check if we've reached the limit
                if max_messages and self.messages_processed >= max_messages:
                    logger.info(f"\n✅ Reached maximum message limit ({max_messages})")
                    break
                    
        except KeyboardInterrupt:
            logger.info("\n⚠️  Consumer stopped by user")
        finally:
            self._print_statistics()
            self.close()
            
    def _print_statistics(self):
        """Print processing statistics."""
        elapsed = time.time() - self.start_time
        rate = self.messages_processed / elapsed if elapsed > 0 else 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 PROCESSING STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"  Messages processed: {self.messages_processed}")
        logger.info(f"  Predictions made: {self.predictions_made}")
        logger.info(f"  Errors: {self.errors}")
        logger.info(f"  Duration: {elapsed:.2f} seconds")
        logger.info(f"  Throughput: {rate:.2f} messages/sec")
        logger.info(f"{'='*60}\n")
        
    def close(self):
        """Close the Kafka consumer."""
        if self.consumer:
            self.consumer.close()
            logger.info("✅ Consumer closed")


def main():
    """Main function to run the consumer."""
    # Configuration
    BOOTSTRAP_SERVERS = 'localhost:9092'
    TOPIC = 'climate-data'
    MODEL_PATH = 'models/baselines/linear_model_sklearn.pkl'
    SCALER_PATH = 'data/processed/splits/scaler_new.pkl'
    IMPUTER_PATH = 'data/processed/splits/imputer_new.pkl'
    MAX_MESSAGES = 10  # Process first 10 messages for demo
    
    try:
        # Initialize predictor
        predictor = ClimatePredictor(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            topic=TOPIC,
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            imputer_path=IMPUTER_PATH
        )
        
        # Start processing
        predictor.run(max_messages=MAX_MESSAGES)
        
        logger.info("✅ Consumer completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Consumer failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
