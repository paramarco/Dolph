# Use Ubuntu as base image
FROM ubuntu:22.04

# Set environment variables to configure tzdata non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
ENV PATH="/opt/conda/bin:$PATH"

# Install the dependencies listed in apt-packages.txt
COPY apt-packages.txt /tmp/apt-packages.txt
RUN apt-get update && \
    xargs apt-get install -y < /tmp/apt-packages.txt && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda init bash && \
    /opt/conda/bin/conda config --set auto_activate_base false

# Update PATH for Miniconda
ENV PATH="/opt/conda/bin:$PATH"

# Create a conda environment with Python 3.7.0
RUN /opt/conda/bin/conda create -y -n py37 python=3.7.0 && \
    /opt/conda/bin/conda clean -afy

# Activate the environment by default
RUN echo "source activate py37" >> ~/.bashrc

# Create an additional virtual environment using python3.7
RUN /opt/conda/envs/py37/bin/python3.7 -m venv /opt/venv

# Upgrade pip to the latest version
RUN /opt/venv/bin/pip install --upgrade pip

# Copy the requirements.txt file to create user and database
COPY requirements.txt /usr/local/bin/requirements.txt

# Install specific versions of Python packages in the virtual environment
RUN /opt/venv/bin/pip install -r /usr/local/bin/requirements.txt

# Add a user with a specified home directory
RUN groupadd -r dolph_user && useradd -m -r -g dolph_user -d /home/dolph_user dolph_user

# Copy all the files from the directory image to the new user's home directory
COPY . /home/dolph_user/

# Debugging: List files in the home directory after copy
RUN ls -lart /home/dolph_user/

# Set ownership of the home directory
RUN chown -R dolph_user:dolph_user /home/dolph_user

# Define the dolph_user data volume
VOLUME /home/dolph_user/

# Define the PostgreSQL data volume
VOLUME /var/lib/postgresql/14/main

# Copy entrypoint script and make it executable
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Expose necessary ports
EXPOSE 5432

