FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# Install some basic utilities
COPY apt-sources.list /etc/apt/sources.list
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    gcc \
    g++ \
    libusb-1.0-0 \
    libgl1-mesa-glx \
    libglib2.0-dev \
    openssh-server \
    openssh-client \
    iputils-ping \
    unzip \
    cmake \
    libssl-dev \
    libosmesa6-dev \
    freeglut3-dev \
    ffmpeg \
    iputils-ping \
 && rm -rf /var/lib/apt/lists/*

# For the convenience for users in China mainland
RUN pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple \
  && export PATH="/usr/local/bin:$PATH" \
  && /bin/bash -c "source ~/.bashrc"
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \
 && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
 && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
 && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ \
 && conda config --set show_channel_urls yes

# Build headless Open3D
RUN apt-get install gpg wget \
 && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
 && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null \
 && apt-get update \
 && rm /usr/share/keyrings/kitware-archive-keyring.gpg \
 && apt-get install kitware-archive-keyring \
 && sudo apt-get install -y cmake
# Clone Open3D repository
RUN git clone --recursive https://github.com/intel-isl/Open3D.git ./Open3D
# Build Open3D with OSMesa
RUN cd ./Open3D \
 && mkdir build \
 && cd build \
 && cmake -DENABLE_HEADLESS_RENDERING=ON -DBUILD_GUI=OFF -DUSE_SYSTEM_GLEW=OFF -DUSE_SYSTEM_GLFW=OFF .. \
 && make -j$(nproc) \
 && make install-pip-package


# Install dependencies
COPY requirements.txt requirements.txt 
RUN pip install -r requirements.txt --user 


# Install svox2
RUN git clone https://github.com/sxyu/svox2.git ./svox2
RUN cd ./svox2 \
 && export TORCH_CUDA_ARCH_LIST=7.0+PTX \
 && python setup.py develop



CMD ["python3"]
