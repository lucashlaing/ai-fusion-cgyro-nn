FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/minimal

USER root

# Install dependency (You may add other dependencies here)
RUN apt update && apt install -y make rsync git vim
RUN wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_linux_amd64.deb && dpkg -i s5cmd_2.2.2_linux_amd64.deb && rm s5cmd_2.2.2_linux_amd64.deb

# Add ssh key
RUN mkdir -p /root/.ssh
ADD .ssh/id_rsa /root/.ssh/id_rsa
ADD .ssh/known_hosts /root/.ssh/known_hosts
RUN chmod 400 /root/.ssh/id_rsa
RUN chmod 400 /root/.ssh/known_hosts

# Pull the latest project
WORKDIR /root/
RUN git clone --depth=1 --branch lucas_runs git@github.com:lucashlaing/ai-fusion-cgyro-nn.git
WORKDIR /root/ai-fusion-cgyro-nn/

# Handle git submodule
RUN git submodule update --init --recursive

# Install conda environment (lean: python + poetry only; everything else via poetry.lock)
RUN conda env create -n ai-fusion-cgyro-nn --file environment.yml && conda clean -qafy

# Activate the new conda environment
SHELL ["/opt/conda/bin/conda", "run", "-n", "ai-fusion-cgyro-nn", "/bin/bash", "-c"]

# Pre-install torch + CUDA wheels in their own layer so the heaviest blob uploads independently
RUN pip install --no-cache-dir torch==2.7.1

# Project deps via poetry (torch already satisfied above)
RUN poetry install --no-root

# Pip install pyrokinetics (poetry can't resolve it cleanly)
RUN pip install --no-cache-dir pyrokinetics==0.8.0

# Comment to rebuild image 11
