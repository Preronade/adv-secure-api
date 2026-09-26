import os, io, json, hmac, hashlib, time, uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

import boto3, jwt, torch
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from model import SmallCNN

EP = os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")
BUCKET = os.getenv("MODEL_BUCKET", "adv-ml-models")
MAX_BODY = 50_000
s3 = boto3.client("s3", endpoint_url=EP)
sm = boto3.client("secretsmanager", endpoint_url=EP)
state = {}

@asynccontextmanager
async def lifespan(app):
    state["jwt_secret"] = sm.get_secret_value(SecretId="jwt-secret")["SecretString"]
    state["user"] = json.loads(sm.get_secret_value(SecretId="api-user")["SecretString"])
    for name in ("standard", "robust"):
        body = s3.get_object(Bucket=BUCKET, Key=f"models/{name}.pt")["Body"].read()
        m = SmallCNN()
        m.load_state_dict(torch.load(io.BytesIO(body), map_location="cpu", weights_only=True))
        m.eval()
        state[name] = m
    yield

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Adversarially-robust MNIST API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
oauth2 = OAuth2PasswordBearer(tokenUrl="token")

@app.middleware("http")
async def harden(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_BODY:
        return JSONResponse({"detail": "payload too large"}, status_code=413)
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["Content-Security-Policy"] = "default-src 'none'"
    return resp

class PredictIn(BaseModel):
    image: list[list[float]]

    @field_validator("image")
    @classmethod
    def check(cls, v):
        if len(v) != 28 or any(len(r) != 28 for r in v):
            raise ValueError("image must be 28x28")
        if any(not (0.0 <= p <= 1.0) for r in v for p in r):   # also rejects NaN
            raise ValueError("pixels must be in [0, 1]")
        return v

def current_user(token: str = Depends(oauth2)) -> str:
    try:
        payload = jwt.decode(token, state["jwt_secret"], algorithms=["HS256"],
                             options={"require": ["exp", "sub"]})
    except jwt.PyJWTError:
        raise HTTPException(401, "invalid token", headers={"WWW-Authenticate": "Bearer"})
    return payload["sub"]

def audit(event: dict):
    try:
        key = f"audit/{datetime.now(timezone.utc):%Y-%m-%d}/{uuid.uuid4()}.json"
        s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(event).encode())
    except Exception:
        pass   # never let logging break a request

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/token")
@limiter.limit("5/minute")
def token(request: Request, form: OAuth2PasswordRequestForm = Depends()):
    u = state["user"]
    h = hashlib.scrypt(form.password.encode(), salt=bytes.fromhex(u["salt"]), n=2**14, r=8, p=1)
    ok_pw = hmac.compare_digest(h, bytes.fromhex(u["hash"]))
    ok_user = hmac.compare_digest(form.username.encode(), u["username"].encode())
    if not (ok_pw and ok_user):
        audit({"event": "login_failed", "ip": get_remote_address(request), "ts": time.time()})
        raise HTTPException(401, "bad credentials")
    now = int(time.time())
    tok = jwt.encode({"sub": u["username"], "iat": now, "exp": now + 900},
                     state["jwt_secret"], algorithm="HS256")
    return {"access_token": tok, "token_type": "bearer"}

@app.post("/predict")
@limiter.limit("30/minute")
def predict(request: Request, body: PredictIn,
            model: Literal["standard", "robust"] = "robust",
            user: str = Depends(current_user)):
    x = torch.tensor(body.image, dtype=torch.float32)[None, None]
    with torch.no_grad():
        probs = torch.softmax(state[model](x), 1)[0]
        squeezed_label = state[model]((x > 0.5).float()).argmax(1).item()   # 1-bit feature squeezing
    label = int(probs.argmax())
    flagged = squeezed_label != label
    audit({"event": "predict", "user": user, "model": model, "label": label,
           "flagged": flagged, "ip": get_remote_address(request), "ts": time.time()})
    return {"label": label, "confidence": round(float(probs.max()), 4),
            "model": model, "suspicious_input": flagged}