#FROM tensorflow/tensorflow:2.0.0-gpu-py3
FROM tensorflow/tensorflow:2.2.0-gpu
MAINTAINER moono.song "toilety@gmail.com"
RUN apt-get update -y
RUN apt-get install -y build-essential
RUN pip install pillow