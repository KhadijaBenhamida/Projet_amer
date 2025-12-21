"""
Simple Kafka Consumer - Minimal version for testing

This is a basic consumer for testing Kafka connectivity.
"""

import json
import logging
from kafka import KafkaConsumer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run simple Kafka consumer."""
    consumer = KafkaConsumer(
        'climate-data',
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='test-group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        consumer_timeout_ms=10000
    )
    
    logger.info("✅ Connected to Kafka - listening for messages...")
    
    count = 0
    for message in consumer:
        count += 1
        logger.info(f"Message {count}: {list(message.value.keys())[:5]}...")
        
        if count >= 5:
            logger.info("Received 5 messages, stopping.")
            break
    
    consumer.close()

if __name__ == "__main__":
    main()
