# Adversarial ML + Secure FastAPI on LocalStack

A portfolio project combining adversarial machine learning, cloud infrastructure emulation, and DevSecOps practices in one pipeline: an MNIST digit classifier hardened against adversarial attacks, served through a security-focused FastAPI backend, backed by AWS services emulated locally via LocalStack, and validated by an automated CI/CD security pipeline.

## What this demonstrates

- **Adversarial robustness**: a standard CNN and a PGD-adversarially-trained CNN, benchmarked against FGSM and PGD attacks using IBM's Adversarial Robustness Toolbox (ART)
- **Cloud-native design**: models and secrets stored in S3 and Secrets Manager (emulated locally via LocalStack), not baked into the application
- **API security**: JWT auth, rate limiting, strict input validation, security headers, audit logging, and a lightweight adversarial-input detector (feature squeezing)
- **DevSecOps**: a GitHub Actions pipeline running static analysis (Bandit), dependency auditing (pip-audit), secret scanning (Gitleaks), and container image scanning (Trivy) on every push

## Results

| Model | Clean accuracy | FGSM accuracy | PGD accuracy |
|---|---|---|---|
| Standard | 0.975 | 0.411 | 0.072 |
| Adversarially trained (PGD) | 0.971 | 0.843 | 0.732 |

The adversarially trained model retains substantially more accuracy under attack, at a small cost to clean accuracy.

## Architecture

- **train.py** trains a standard CNN and a PGD-adversarially-trained CNN using ART, then saves both models and metrics.
- **bootstrap_aws.py** uploads the models and stores JWT/user secrets in **LocalStack** (emulating S3 + Secrets Manager).
- **api.py** (FastAPI) loads the models and secrets from LocalStack at startup, and serves predictions behind JWT auth and rate limiting.
- Clients hit the API via `curl` or the Swagger UI at `/docs`.

## Tech stack

Python 3.12 · PyTorch (CPU) · Adversarial Robustness Toolbox (ART) · FastAPI · Uvicorn · boto3 · LocalStack · Docker · GitHub Actions · Bandit · pip-audit · Gitleaks · Trivy

## Project structureadv-secure-api/
├── src/
│ ├── model.py # CNN architecture
│ ├── train.py # standard + PGD adversarial training, saves metrics
│ ├── bootstrap_aws.py # provisions S3 bucket + secrets in LocalStack
│ ├── api.py # FastAPI service (auth, rate limiting, predict endpoint)
│ └── make_sample.py # crafts a real adversarial MNIST digit for testing
├── tests/test_basic.py # unit tests
├── Dockerfile
├── docker-compose.yml # LocalStack service
├── .github/workflows/ci.yml # CI/CD security pipeline
└── requirements*.txt

## Running it locally

Prerequisites: Docker, Python 3.12, a free LocalStack account (Hobby plan — no card required).

```bash
# 1. Start LocalStack
docker compose up -d
set -a; source .env; set +a

# 2. Train both models
python src/train.py

# 3. Push models + secrets into LocalStack
python src/bootstrap_aws.py

# 4. Run the API
uvicorn --app-dir src api:app --host 127.0.0.1 --port 8000

# 5. In a second terminal — craft an adversarial sample and test both models
python src/make_sample.py
TOKEN=$(curl -s -X POST localhost:8000/token -d "username=analyst&password=$API_PASSWORD" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
curl -s -X POST "localhost:8000/predict?model=robust" -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d @samples/adv.json
```

Or run it fully containerized:
```bash
docker build -t adv-api .
docker run --rm --network host -e AWS_ENDPOINT_URL -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION adv-api
```

## CI/CD pipeline

Every push runs:
1. **Bandit** — static analysis for insecure code patterns
2. **pip-audit** — known CVEs in Python dependencies
3. **pytest** — unit tests
4. **Gitleaks** — scans for committed secrets
5. **Docker build**
6. **Trivy** — container image vulnerability scan (fails the build on CRITICAL/HIGH findings)

The pipeline itself caught and required fixing three real CVEs (`setuptools`, `wheel`, `jaraco.context`) and a transitive one (`msgpack`) in the base image dependencies before the build was allowed to pass — a working example of shift-left security in practice.

## Known limitations (intentional, explored further in a companion OWASP API audit)

- No TLS (local dev only)
- JWT uses HS256 with a shared secret; no token revocation
- Feature-squeezing detector is a weak, non-adaptive defense
- Rate limiting is per-IP and in-memory only

## Author

Prerona De — B.Tech CSE (AI-driven DevOps), Jain University, Bangalore