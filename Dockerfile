# Use an official Ubuntu image as the base image
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Set the working directory
WORKDIR /

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    tzdata \
    swig \
    build-essential \
    gcc \
    g++ \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && apt-get clean

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    /bin/bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

# Update PATH to include conda
ENV PATH=/opt/conda/bin:$PATH

# RUN conda install -y python=3.8 && \
#     conda install -y -c pytorch pytorch torchvision

# Copy the current directory contents into the Docker image at /app
#COPY . .

# Clone the repository during the build process
RUN git clone https://github.com/roihezkiyahu/RLAlgorithms.git

RUN conda env create -f /RLAlgorithms/environment.yml || true

RUN conda install -y python=3.8 && \
    conda install -y -c pytorch pytorch torchvision imageio numpy matplotlib  pandas  tqdm pyyaml && \
    conda install -y opencv=4.6

RUN pip install pygame \
    imageio \
    swig \
    "gym[box2d]" \
    "gym[atari]" \
    "autorom[accept-rom-license]" \
    ale_py \
    gymnasium \
    gymnasium[atari] \
    google-cloud-storage

ENV GOOGLE_APPLICATION_CREDENTIALS="/RLAlgorithms/trainer/service-account-key.json"

# RUN AutoRom

# Activate the Conda environment
ENV PATH /opt/conda/envs/RLAlgorithms/bin:$PATH

# Make sure the environment is activated
RUN echo "source activate RLAlgorithms" > ~/.bashrc

# Sets up the entry point to invoke the trainer.
CMD ["sh", "-c", "git pull && python -m RLAlgorithms.trainer.task"]
