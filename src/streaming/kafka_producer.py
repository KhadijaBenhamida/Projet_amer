"""
Kafka Producer for Climate Data Streaming

This module implements a Kafka producer that reads climate data from parquet files
and streams it to a Kafka topic for real-time processing.
"""

import pandas as pd
import json
import time
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClimateDataProducer:
    """
    Kafka producer for streaming climate data.
    
    Attributes:
        bootstrap_servers (str): Kafka broker address
        topic (str): Kafka topic name
        producer (KafkaProducer): Kafka producer instance
    """
    
    def __init__(self, bootstrap_servers='localhost:9092', topic='climate-data'):
        """
        Initialize the Climate Data Producer.
        
        Args:
            bootstrap_servers (str): Kafka broker address
            topic (str): Kafka topic to send data to
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None
        self._connect()
        
    def _connect(self):
        """Establish connection to Kafka broker."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1
            )
            logger.info(f"✅ Connected to Kafka broker at {self.bootstrap_servers}")
        except KafkaError as e:
            logger.error(f"❌ Failed to connect to Kafka: {e}")
            raise
            
    def load_data(self, data_path):
        """
        Load climate data from parquet file.
        
        Args:
            data_path (str): Path to the parquet file
            
        Returns:
            pd.DataFrame: Loaded climate data
        """
        try:
            df = pd.read_parquet(data_path)
            logger.info(f"✅ Loaded {len(df)} records from {data_path}")
            logger.info(f"   Features: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
            
    def send_data(self, df, delay=0.1, max_records=None):
        """
        Send climate data records to Kafka topic.
        
        Args:
            df (pd.DataFrame): Climate data to send
            delay (float): Delay between messages in seconds
            max_records (int): Maximum number of records to send (None = all)
        """
        if max_records:
            df = df.head(max_records)
            
        total_records = len(df)
        logger.info(f"🚀 Starting to send {total_records} records to topic '{self.topic}'")
        
        sent_count = 0
        error_count = 0
        start_time = time.time()
        
        for idx, row in df.iterrows():
            try:
                # Convert row to dictionary
                record = row.to_dict()
                
                # Handle NaN values
                record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
                
                # Add metadata
                record['_timestamp'] = time.time()
                record['_index'] = int(idx) if isinstance(idx, (int, float)) else str(idx)
                
                # Send to Kafka
                future = self.producer.send(
                    self.topic,
                    key=str(idx),
                    value=record
                )
                
                # Wait for message to be sent
                future.get(timeout=10)
                sent_count += 1
                
                # Progress logging
                if sent_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = sent_count / elapsed if elapsed > 0 else 0
                    logger.info(f"  📊 Sent {sent_count}/{total_records} records ({rate:.2f} msg/sec)")
                
                # Rate limiting
                if delay > 0:
                    time.sleep(delay)
                    
            except Exception as e:
                error_count += 1
                logger.error(f"  ❌ Error sending record {idx}: {e}")
                
        # Final statistics
        elapsed = time.time() - start_time
        rate = sent_count / elapsed if elapsed > 0 else 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📈 STREAMING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"  Total records: {total_records}")
        logger.info(f"  Successfully sent: {sent_count}")
        logger.info(f"  Errors: {error_count}")
        logger.info(f"  Duration: {elapsed:.2f} seconds")
        logger.info(f"  Throughput: {rate:.2f} messages/sec")
        logger.info(f"{'='*60}\n")
        
    def close(self):
        """Close the Kafka producer connection."""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("✅ Kafka producer closed")


def main():
    """Main function to run the Kafka producer."""
    # Configuration
    DATA_PATH = Path('data/processed/features_data.parquet')
    BOOTSTRAP_SERVERS = 'localhost:9092'
    TOPIC = 'climate-data'
    DELAY = 0.002  # 2ms between messages (500 msg/sec)
    MAX_RECORDS = 10000  # Send first 10,000 records
    
    try:
        # Initialize producer
        producer = ClimateDataProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            topic=TOPIC
        )
        
        # Load data
        df = producer.load_data(DATA_PATH)
        
        # Send data
        producer.send_data(df, delay=DELAY, max_records=MAX_RECORDS)
        
        # Cleanup
        producer.close()
        
        logger.info("✅ Producer completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Producer stopped by user")
        if 'producer' in locals():
            producer.close()
    except Exception as e:
        logger.error(f"❌ Producer failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
