FROM python:3.8.5
# Image from dockerhub

ENV PYTHONUNBUFFERED 1 
ENV PYTHONPATH=/app
EXPOSE 8000 
# Expose the port 8000 in which our application runs
WORKDIR /app 
# Make /app as a working directory in the container
# Copy requirements from host, to docker container in /app 
COPY ./requirements.txt .
# Copy everything from ./src directory to /app in the container
COPY ./tweet ./tweet
RUN apt-get update \
    && apt-get install python3-dev python3-pip -y \
    && python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt
# Run the application in the port 8000
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "tweet.api:app"]
