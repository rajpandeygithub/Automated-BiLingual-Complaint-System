# Use the official Python image as the base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY frontend_requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r frontend_requirements.txt

# Copy the application code into the container
COPY . .

# Copy the .env file into the container
COPY .env .env

# Expose port 8080 (required by Cloud Run)
EXPOSE 8080

# Command to run the Streamlit application with the expanded PORT variable
CMD ["sh", "-c", "streamlit run frontend.py --server.port=$PORT --server.address=0.0.0.0"]
