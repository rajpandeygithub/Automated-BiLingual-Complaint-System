# Use the official Python image as the base
FROM python:3.9-slim

ENV PROJECT_ID=bilingualcomplaint-system
ENV LOCATION=us-east1

ENV DRIFT_EN_URL=https://us-east1-bilingualcomplaint-system.cloudfunctions.net/data_drift
ENV DRIFT_HI_URL=https://us-east1-bilingualcomplaint-system.cloudfunctions.net/data-drift-hindi

ENV PRODUCT_ENDPOINT_ID=5930875671386521600
ENV DEPARTMENT_ENDPOINT_ID=7734848797125115904
#7987050376257863680

ENV DATASET_ID=MLOps
ENV PREDICTION_TABLE_ID=predicted_records

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose port 80 to the outside world
EXPOSE 8080

# Command to run application
CMD ["python", "server.py"]