FROM  gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/minimal

USER root

# Install dependency (You may add other dependencies here)
RUN apt update && apt install -y make rsync git vim
RUN wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_linux_amd64.deb && dpkg -i s5cmd_2.2.2_linux_amd64.deb && rm s5cmd_2.2.2_linux_amd64.deb

# Add ssh key
RUN mkdir -p /root/.ssh
ADD .ssh/id_rsa /root/.ssh/id_rsa
ADD /.ssh/known_hosts /root/.ssh/known_hosts

# Create known_hosts
RUN ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts

# Remove host checking
RUN echo "Host github.com\n\tStrictHostKeyChecking no\n" >> /root/.ssh/config
# Update permissions

RUN chmod 400 /root/.ssh/id_rsa
RUN chmod 400 /root/.ssh/config
RUN chmod 400 /root/.ssh/known_hosts

# Pull the latest project
WORKDIR /root/
RUN git clone --depth=1 https://github.com/lucashlaing/ai-fusion-cgyro-nn.git --branch online
WORKDIR /root/ai-fusion-cgyro-nn/

# Handle git submodule
RUN git submodule update --init --recursive

# Install conda environment
RUN conda update --all
RUN conda env create -n ai-fusion-cgyro-nn --file environment.yml
RUN conda clean -qafy

# Activate the new conda environment and install poetry
SHELL ["/opt/conda/bin/conda", "run", "-n", "PROJECT_NAME", "/bin/bash", "-c"]
RUN poetry lock
RUN poetry install --no-root
#Pip install as the poetry install is broken due to their dependency issues
RUN pip install pyrokinetics