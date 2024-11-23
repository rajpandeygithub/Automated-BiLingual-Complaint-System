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

# Expose the default Streamlit port
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
