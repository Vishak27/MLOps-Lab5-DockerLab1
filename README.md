# Wine Classification with Logistic Regression

This Docker lab trains a Logistic Regression model on the Wine dataset with comprehensive evaluation and artifact saving.

## Features
- Wine dataset (178 samples, 13 features, 3 classes)
- Data preprocessing with StandardScaler
- Logistic Regression classifier
- Comprehensive evaluation metrics
- Organized output structure (models + results)
- Model persistence and artifact saving
- Configurable hyperparameters via environment variables

## Build the Docker image
```bash
docker build -t dockerfile:v1 .
```

## Run the Docker container

### Basic run (default parameters)
```bash
docker run dockerfile:v1
```

### Run with volume mounting to extract artifacts
```bash
docker run -v $(pwd)/outputs:/app/outputs dockerfile:v1
```

After running, you'll have an `outputs/` folder in your directory with all artifacts.

### Run with custom hyperparameters
```bash
docker run -e MAX_ITER=2000 -e SOLVER=saga -e TEST_SIZE=0.3 dockerfile:v1
```

### Run with both volume mounting and custom parameters
```bash
docker run -v $(pwd)/outputs:/app/outputs -e MAX_ITER=2000 -e TEST_SIZE=0.25 dockerfile:v1
```

## Environment Variables
- `TEST_SIZE`: Test set size (default: 0.2)
- `RANDOM_STATE`: Random seed (default: 42)
- `MAX_ITER`: Maximum iterations for solver (default: 1000)
- `SOLVER`: Optimization algorithm (default: lbfgs, options: lbfgs, saga, newton-cg, sag)


## Extracting Artifacts from Container

To get the trained model and results on your local machine:
```bash
# Run with volume mounting
docker run -v $(pwd)/outputs:/app/outputs dockerfile:v1

# Check the outputs folder
ls -R outputs/
```

## Clean Up
```bash
# Remove stopped containers
docker container prune

# Remove the image
docker rmi dockerfile:v1

# Remove outputs folder
rm -rf outputs/
```

(Note: This lab assignment is a modification of the original lab from "ML with Ramim" where the work was on Random Forest for iris datset - here the work is done on Logistic Regression for Wine Quality dataset)
