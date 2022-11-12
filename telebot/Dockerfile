FROM python:3.8-slim

ARG service_home="/home/Telebot"

# Configure apt and install packages
RUN apt-get update -y && \
    apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dev \
    git \
    ffmpeg \
    # cleanup
    && apt-get autoremove -y \
    && apt-get clean -y \
    && apt-get install espeak -y\
    && rm -rf /var/lib/apt/lists

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

RUN mkdir "$service_home" \processing \
    && mkdir "$service_home/files"

ADD bot.py "$service_home/bot.py"
ADD processing.py "$service_home/processing.py"

WORKDIR $service_home/

ENTRYPOINT ["python3", "bot.py"]