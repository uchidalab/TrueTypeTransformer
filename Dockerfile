FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

# LABEL maintainer="example@example.com"

# Timezone setting
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

# Install something
RUN apt-get update && apt-get install -y --no-install-recommends bash curl fish git nano sudo

RUN rm /usr/bin/python3
RUN rm /usr/bin/python3.8

# OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libopencv-dev

# Install Python
ENV PYTHON_VERSION 3.9
RUN apt-get update && apt-get install -y --no-install-recommends python${PYTHON_VERSION}

# Add User & Group
ARG UID
ARG USER
ARG PASSWORD
RUN groupadd -g 1000 ${USER}_group
RUN useradd -m --uid=${UID} --gid=${USER}_group --groups=sudo ${USER}
RUN echo ${USER}:${PASSWORD} | chpasswd
RUN echo 'root:root' | chpasswd

ENV PATH ${PATH}:/home/${USER}/.local/bin

# Change working directory
ENV WORK_DIR /workspace
RUN mkdir ${WORK_DIR}
RUN chown ${USER}:${USER}_group ${WORK_DIR}
WORKDIR ${WORK_DIR}

# Change User
USER ${USER}

# Install pip
RUN curl --silent https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python${PYTHON_VERSION} get-pip.py
RUN rm get-pip.py
RUN pip install --upgrade pip

# Install Python library
COPY requirements.txt /
RUN pip install -r /requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install flake8 autopep8

# or 
# RUN pip install hydra-core --upgrade
# RUN pip install TTFQuery==2.0.0b1 pandas matplotlib torchinfo scikit-learn jupyterlab tensorboard tqdm einops seaborn
# RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
