FROM pytorch/pytorch

# if you forked EasyOCR, you can pass in your own GitHub username to use your fork
# i.e. gh_username=myname
ARG gh_username=JaidedAI
ARG service_home="/home/EasyOCR"

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
    && rm -rf /var/lib/apt/lists

# Clone EasyOCR repo
RUN mkdir "$service_home" \
    && git clone "https://github.com/$gh_username/EasyOCR.git" "$service_home" \
    && cd "$service_home" \
    && git remote add upstream "https://github.com/JaidedAI/EasyOCR.git" \
    && git pull upstream master

RUN pip3 install flask
EXPOSE 8080

# Build
RUN cd "$service_home" \
    && python setup.py build_ext --inplace -j 4 \
    && python -m pip install -e .

# Add pretrained model
ADD ./pretrained_models/ /home/.EasyOCR/model
ADD ocr_reader.py /home/ocr_reader.py

WORKDIR /home/

ENTRYPOINT ["python3", "ocr_
+reader.py"]