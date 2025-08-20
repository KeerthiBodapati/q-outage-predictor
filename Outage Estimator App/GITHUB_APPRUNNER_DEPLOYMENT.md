# GitHub to AWS App Runner Deployment Guide

## Step 1: Prepare Your Repository

### Required Files in Root Directory:
- `app.py` (your Streamlit app)
- `apprunner.yaml` (App Runner configuration)
- `requirements_streamlit.txt` (dependencies)
- `artifacts/` folder (model files)
- `data/` folder (training data)

## Step 2: Push to GitHub

```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit: Outage Predictor App"

# Create GitHub repository and push
git remote add origin https://github.com/YOUR_USERNAME/outage-predictor.git
git branch -M main
git push -u origin main
```

## Step 3: Deploy to App Runner

### Via AWS Console:
1. Go to **AWS App Runner Console**
2. Click **"Create service"**
3. Choose **"Source code repository"**
4. Click **"Add new"** to connect GitHub
5. Authorize AWS App Runner to access your GitHub
6. Select your repository: `outage-predictor`
7. Branch: `main`
8. **Configuration**: Use configuration file (`apprunner.yaml`)
9. Service name: `outage-predictor`
10. Click **"Create & deploy"**

### Via AWS CLI:
```bash
aws apprunner create-service \
  --service-name outage-predictor \
  --source-configuration '{
    "CodeRepository": {
      "RepositoryUrl": "https://github.com/YOUR_USERNAME/outage-predictor",
      "SourceCodeVersion": {
        "Type": "BRANCH",
        "Value": "main"
      },
      "CodeConfiguration": {
        "ConfigurationSource": "CONFIGURATION_FILE"
      }
    },
    "AutoDeploymentsEnabled": true
  }'
```

## Step 4: Monitor Deployment

```bash
# Check service status
aws apprunner describe-service --service-arn YOUR_SERVICE_ARN

# View logs
aws logs tail /aws/apprunner/outage-predictor --follow
```

## Step 5: Access Your App

Your app will be available at:
`https://YOUR_SERVICE_ID.us-east-1.awsapprunner.com`

## Auto-Deployment

Every push to `main` branch will automatically trigger a new deployment.

## Cost Estimate

- **Provisioned instances**: $0.007/hour per instance
- **Requests**: $0.000002 per request
- **Data transfer**: $0.09/GB

**Example**: ~$5-15/month for moderate usage

## Troubleshooting

### Build Fails:
- Check `requirements_streamlit.txt` has all dependencies
- Verify `apprunner.yaml` syntax
- Check GitHub repository is public or properly connected

### App Won't Start:
- Ensure port 8080 in `apprunner.yaml`
- Check Streamlit runs locally first
- Review CloudWatch logs

### Common Issues:
```bash
# If build fails, check requirements
pip install -r requirements_streamlit.txt
streamlit run app.py --server.port=8080

# Test locally first
docker build -f Dockerfile.apprunner -t test-app .
docker run -p 8080:8080 test-app
```

## Management Commands

```bash
# Update service
aws apprunner start-deployment --service-arn YOUR_SERVICE_ARN

# Pause service (stop billing)
aws apprunner pause-service --service-arn YOUR_SERVICE_ARN

# Resume service
aws apprunner resume-service --service-arn YOUR_SERVICE_ARN

# Delete service
aws apprunner delete-service --service-arn YOUR_SERVICE_ARN
```

Your Streamlit app will be live with HTTPS, auto-scaling, and automatic deployments!