import os
import boto3
import pandas as pd

# Get S3 configuration from environment
bucket = os.environ['S3_BUCKET']
key = os.environ['S3_KEY']

# Create data directory if missing
os.makedirs('./data', exist_ok=True)

# Download from S3
s3 = boto3.client('s3')
local_path = './data/raw.csv'
s3.download_file(bucket, key, local_path)

# Load with pandas and print results
df = pd.read_csv(local_path)
print(f"Saved to: {local_path}")
print(f"Row count: {len(df)}")