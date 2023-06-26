FROM python:3.10.7-slim

WORKDIR /app

COPY requirements.txt /tmp

RUN apt-get update -q && \
    apt-get install --no-install-recommends -qy \
    build-essential


RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt
    