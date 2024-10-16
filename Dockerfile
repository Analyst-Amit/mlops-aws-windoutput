# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files
COPY requirements.txt /app/requirements.txt
COPY src/ /app/src/
COPY config.ini /app/config.ini
COPY conf/ /app/conf/

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/app/src
ENV GIT_PYTHON_REFRESH=quiet

# Use CMD to specify the script and its default arguments
CMD ["python", "/app/src/pipelines/batch_score.py", "--env", "staging"]
