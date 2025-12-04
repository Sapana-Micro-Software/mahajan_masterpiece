# ECG Classification Models - Deployment Guide

## Docker Deployment

### Prerequisites
- Docker (20.10+)
- Docker Compose (1.29+)
- At least 4GB RAM available
- 10GB disk space

### Quick Start

1. **Build the Docker image:**
```bash
docker-compose build
```

2. **Start the services:**
```bash
docker-compose up -d
```

3. **Check service status:**
```bash
docker-compose ps
docker-compose logs -f ecg-api
```

4. **Access the API:**
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health

5. **Stop the services:**
```bash
docker-compose down
```

### Configuration

#### Environment Variables

Edit `docker-compose.yml` to customize:

```yaml
environment:
  - WORKERS=4              # Number of Uvicorn workers
  - PORT=8000             # API port
  - PYTHONUNBUFFERED=1    # Python output buffering
```

#### Resource Limits

Adjust in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
```

### Production Deployment

#### 1. With Nginx Load Balancer

```bash
# Start all services including Nginx
docker-compose up -d

# API accessible via Nginx on port 80
curl http://localhost/health
```

#### 2. Scaling the API

```bash
# Scale to multiple API instances
docker-compose up -d --scale ecg-api=3
```

#### 3. Enable HTTPS

1. Obtain SSL certificates (e.g., Let's Encrypt)
2. Place certificates in `./ssl/` directory
3. Uncomment HTTPS section in `nginx.conf`
4. Restart services: `docker-compose restart nginx`

### API Usage Examples

#### Python
```python
import requests

# Single prediction
response = requests.post(
    'http://localhost:8000/predict',
    json={
        'signal': [0.1, 0.2, ...],  # Your ECG signal
        'sampling_rate': 250
    }
)
result = response.json()
print(f"Prediction: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2f}")
```

#### cURL
```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model/info

# Prediction (with sample data)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"signal": [0.1, 0.2, 0.3, ...], "sampling_rate": 250}'
```

#### JavaScript/Node.js
```javascript
const axios = require('axios');

async function predictECG(signal) {
    const response = await axios.post('http://localhost:8000/predict', {
        signal: signal,
        sampling_rate: 250
    });
    return response.data;
}
```

### Monitoring

#### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f ecg-api

# Last N lines
docker-compose logs --tail=100 ecg-api
```

#### Check health
```bash
docker-compose exec ecg-api curl http://localhost:8000/health
```

### Troubleshooting

#### API not responding
```bash
# Check if container is running
docker-compose ps

# Check logs
docker-compose logs ecg-api

# Restart service
docker-compose restart ecg-api
```

#### Out of memory
```bash
# Increase memory limit in docker-compose.yml
# Or reduce number of workers
```

#### Port already in use
```bash
# Change port in docker-compose.yml
ports:
  - "8080:8000"  # Use 8080 instead of 8000
```

## Kubernetes Deployment (Optional)

### Create Kubernetes manifests

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ecg-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ecg-api
  template:
    metadata:
      labels:
        app: ecg-api
    spec:
      containers:
      - name: ecg-api
        image: ecg-classification-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ecg-api-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: ecg-api
```

### Deploy to Kubernetes
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl get pods
kubectl get services
```

## Cloud Deployment

### AWS (Elastic Beanstalk)
```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init

# Create environment
eb create ecg-api-env

# Deploy
eb deploy
```

### Google Cloud Run
```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/ecg-api

# Deploy
gcloud run deploy ecg-api \
  --image gcr.io/PROJECT_ID/ecg-api \
  --platform managed \
  --memory 4Gi \
  --cpu 2
```

### Azure Container Instances
```bash
# Build image
az acr build --registry myregistry --image ecg-api .

# Deploy
az container create \
  --resource-group mygroup \
  --name ecg-api \
  --image myregistry.azurecr.io/ecg-api \
  --cpu 2 \
  --memory 4 \
  --ports 8000
```

## Performance Optimization

### 1. Use GPU (if available)
```yaml
# docker-compose.yml
services:
  ecg-api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 2. Enable Model Caching
- Pre-load models in startup event
- Use Redis for prediction caching

### 3. Optimize Workers
```bash
# Formula: (2 x CPU cores) + 1
WORKERS = (2 * 4) + 1 = 9
```

## Security Best Practices

1. **Enable authentication:**
   - Add API keys
   - Implement OAuth2
   - Use JWT tokens

2. **Rate limiting:**
   - Already configured in Nginx
   - Adjust as needed

3. **HTTPS only:**
   - Always use TLS in production
   - Redirect HTTP to HTTPS

4. **Input validation:**
   - Already implemented via Pydantic
   - Additional sanitization as needed

5. **Monitoring:**
   - Set up logging aggregation
   - Use Prometheus/Grafana
   - Alert on errors

## Maintenance

### Update models
```bash
# Copy new model
docker cp model.pth ecg-api:/app/models/

# Restart service
docker-compose restart ecg-api
```

### Backup
```bash
# Backup models
docker cp ecg-api:/app/models ./backup/models/

# Backup logs
docker cp ecg-api:/app/logs ./backup/logs/
```

### Updates
```bash
# Pull latest code
git pull

# Rebuild
docker-compose build

# Restart
docker-compose up -d
```

## Support

For issues or questions:
- Check logs: `docker-compose logs`
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
