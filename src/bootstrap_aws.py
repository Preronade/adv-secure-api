import os, json, hashlib, secrets
import boto3
from botocore.exceptions import ClientError

ep = os.environ["AWS_ENDPOINT_URL"]
s3 = boto3.client("s3", endpoint_url=ep)
sm = boto3.client("secretsmanager", endpoint_url=ep)
BUCKET = "adv-ml-models"

try:
    s3.create_bucket(Bucket=BUCKET)
except ClientError as e:
    if e.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
        raise
s3.put_bucket_versioning(Bucket=BUCKET, VersioningConfiguration={"Status": "Enabled"})
s3.put_public_access_block(Bucket=BUCKET, PublicAccessBlockConfiguration={
    k: True for k in ("BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets")})
s3.put_bucket_encryption(Bucket=BUCKET, ServerSideEncryptionConfiguration={
    "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]})

for f in ("standard.pt", "robust.pt", "metrics.json"):
    s3.upload_file(f"models/{f}", BUCKET, f"models/{f}")

def put_secret(name, value):
    try:
        sm.create_secret(Name=name, SecretString=value)
    except sm.exceptions.ResourceExistsException:
        sm.put_secret_value(SecretId=name, SecretString=value)

salt = os.urandom(16)
pw_hash = hashlib.scrypt(os.environ["API_PASSWORD"].encode(), salt=salt, n=2**14, r=8, p=1)
put_secret("jwt-secret", secrets.token_hex(32))
put_secret("api-user", json.dumps({"username": "analyst", "salt": salt.hex(), "hash": pw_hash.hex()}))
print("bootstrap done")