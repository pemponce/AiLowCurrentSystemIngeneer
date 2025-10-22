import os
from minio import Minio

_client = None


def get_client():
    global _client
    if _client is None:
        endpoint = os.getenv('S3_ENDPOINT', 'http://minio:9000').replace('http://', '').replace('https://', '')
        access = os.getenv('S3_ACCESS_KEY', 'minio')
        secret = os.getenv('S3_SECRET_KEY', 'minio12345')
        secure = os.getenv('S3_ENDPOINT', 'http://minio:9000').startswith('https://')
        _client = Minio(endpoint, access_key=access, secret_key=secret, secure=secure)
    return _client


def upload_file(bucket: str, src_path: str, dst_key: str):
    c = get_client()
    if not c.bucket_exists(bucket):
        c.make_bucket(bucket)
    c.fput_object(bucket, dst_key, src_path)
    return f"s3://{bucket}/{dst_key}"
