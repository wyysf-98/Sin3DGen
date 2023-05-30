## Build Docker Environment and use with GPU Support

Before you can use this Docker environment, you need to have the following:

- Docker installed on your system
- NVIDIA drivers installed on your system
- NVIDIA Container Toolkit installed on your system

### Change the GPU architecture in Dockerfile

You shold change the TORCH_CUDA_ARCH_LIST accoording to your [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) in Dockerfile before building the image.
For example, if you are using an NVIDIA V100 GPU, you should set the `TORCH_CUDA_ARCH_LIST` environment variable in `Dockerfile` to the following value:
   ```sh
   export TORCH_CUDA_ARCH_LIST=7.0+PTX
   ```

### Build and Run
1. Build docker image:
   ```sh
   docker build -t Sin3DGen:latest .
   ```
2. Start the docker container:
   ```sh
   docker run --gpus all -it Sin3DGen:latest /bin/bash
   ```
3. Clone the repository:
   ```sh
   git clone git@github.com:wyysf-98/Sin3DGen.git
   ```

## Troubleshooting

If you encounter any issues with the Docker environment with GPU support, please check the following:

- Make sure that you have installed the NVIDIA drivers and NVIDIA Container Toolkit on your system.
- Make sure that you have specified the --gpus all option when starting the Docker container.
- Make sure that your deep learning application is configured to use the GPU.