"""
Spark Structured Streaming Consumer for Climate Data

NOTE: This requires PySpark which has compatibility issues with Python 3.13 on Windows.
Use demo_consumer.py instead for Windows environments.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define schema for climate data
schema = StructType([
    StructField("temperature", DoubleType()),
    StructField("T_max", DoubleType()),
    StructField("T_min", DoubleType()),
    # Add other fields as needed
])

def create_spark_session():
    """Create and configure Spark session."""
    return (SparkSession.builder
            .appName("ClimateDataStreaming")
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0")
            .config("spark.sql.streaming.schemaInference", "true")
            .getOrCreate())

def main():
    """Run Spark Structured Streaming consumer."""
    logger.info("Starting Spark Streaming Consumer...")
    
    spark = create_spark_session()
    
    # Read from Kafka
    df = (spark
          .readStream
          .format("kafka")
          .option("kafka.bootstrap.servers", "localhost:9092")
          .option("subscribe", "climate-data")
          .load())
    
    # Parse JSON
    parsed_df = df.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")
    
    # Write to console for demo
    query = (parsed_df
             .writeStream
             .format("console")
             .outputMode("append")
             .start())
    
    query.awaitTermination()

if __name__ == "__main__":
    main()
