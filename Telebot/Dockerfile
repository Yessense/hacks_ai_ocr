FROM pytorch/pytorch

# if you forked EasyOCR, you can pass in your own GitHub username to use your fork
# i.e. gh_username=myname
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
    # cleanup
    && apt-get autoremove -y \
    && apt-get clean -y \
    && apt-get install espeak -y\
    && rm -rf /var/lib/apt/lists

# Clone EasyOCR repo
RUN mkdir "$service_home" \
    && mkdir "$service_home/files"

RUN pip3 install flask \
    && pip3 install pyTelegramBotAPI \
    && pip3 install pyttsx3 \
    && pip3 install opencv-python

EXPOSE 8083


# Add pretrained model
ADD bot.py "$service_home/bot.py"
ADD processing.py "$service_home/processing.py"

WORKDIR $service_home/

ENTRYPOINT ["python3", "bot.py"]