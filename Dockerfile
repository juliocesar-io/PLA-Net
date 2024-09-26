FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update && apt-get install -y libxrender1 build-essential
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
RUN pip install torch-scatter==2.0.9  torch-cluster==1.6.0 torch-spline-conv==1.2.1 torch-geometric==2.1.0 -f https://data.pyg.org/whl/torch-1.13.1+cu116.html

# Create a new user named "user" with UID 1000
RUN useradd -m -u 1000 user

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    SYSTEM=spaces \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's app directory as root
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app
COPY . $HOME/app

# Change ownership of the app directory to "user"
RUN chown -R user:user $HOME/app

# Switch to the "user" user
USER user

# Upgrade pip as the user
RUN pip install --no-cache-dir --upgrade pip

# Install the local package as the user
RUN pip install --user .

# Set the default command to bash
CMD ["/bin/bash"]