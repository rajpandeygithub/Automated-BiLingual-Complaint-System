# Use a base Python image
FROM python:3.10-slim

# Copy the entire pipeline folder into the container
COPY pipeline /app/pipeline

# Set the working directory
WORKDIR /app/pipeline

# Set environment variable for Google authentication
ENV GOOGLE_APPLICATION_CREDENTIALS "/app/pipeline/all_in_one_service_account_key.json"

# Install dependencies
RUN pip install -r pipeline_requirements.txt

# Run the pipeline script
CMD ["python", "final_pipeline.py"]
