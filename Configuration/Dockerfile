# Use Ubuntu as base image
FROM ubuntu:22.04

# Set environment variables to configure tzdata non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
ENV PATH="/opt/conda/bin:$PATH"
# Set the DISPLAY environment variable
ENV DISPLAY=:1
# Verify Java installation and set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
# Set environment variables for TWS
ENV TWS_VERSION=latest
ENV TWS_INSTALLER_URL="https://download2.interactivebrokers.com/installers/tws/latest/tws-latest-linux-x64.sh"
ENV TWS_INSTALL_DIR="/root/Jts"
# Set environment variables for PostgreSQL
ENV PGDATA /var/lib/postgresql/14/main
ENV PATH /usr/lib/postgresql/14/bin:$PATH
# Update PATH for Miniconda
ENV PATH="/opt/conda/bin:$PATH"

# Install the dependencies listed in apt-packages.txt
COPY new-apt-packages.txt /tmp/new-apt-packages.txt
RUN apt-get update && \
    xargs apt-get install -y < /tmp/new-apt-packages.txt && \
    rm -rf /var/lib/apt/lists/*
 
# Add commands to configure dbus and the GNOME environment   
RUN mkdir -p /var/run/dbus && \
    dbus-uuidgen > /var/lib/dbus/machine-id


# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda init bash && \
    /opt/conda/bin/conda config --set auto_activate_base false

# Create a conda environment with Python 3.7.0
RUN /opt/conda/bin/conda create -y -n py37 python=3.7.0 && \
    /opt/conda/bin/conda clean -afy

# Ensure bash is explicitly called
SHELL ["/bin/bash", "-c"]

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
RUN groupadd -r dolph_user && useradd -m -r -g dolph_user -d /home/dolph_user -s /bin/bash dolph_user

# Copy all the files from the directory image to the new user's home directory
COPY . /home/dolph_user/

# Debugging: List files in the home directory after copy
RUN ls -lart /home/dolph_user/

# Set ownership of the home directory
RUN chown -R dolph_user:dolph_user /home/dolph_user

# Add dolph_user to sudo group
RUN usermod -aG sudo dolph_user

# Configure sudoers to allow passwordless sudo for dolph_user
RUN echo "dolph_user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Define the dolph_user data volume
VOLUME /home/dolph_user/

# Define the PostgreSQL data volume
VOLUME /var/lib/postgresql/14/main

# Download IB Gateway installation script
ADD https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh /opt/ibgateway-installer.sh
RUN chmod +x /opt/ibgateway-installer.sh

# Copy entrypoint script and make it executable
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/bin/bash", "/usr/local/bin/entrypoint.sh"]

# Keep the container alive by running bash as the final command
CMD ["bash"]