# Use the official Python slim image for a lightweight container
FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Set the working directory in the container
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy local code to the container image
COPY . ./

# Install production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the web service on container startup
# Use uvicorn to run the FastAPI app, binding to the $PORT environment variable
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT
