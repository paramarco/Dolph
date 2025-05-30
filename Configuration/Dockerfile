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

# Ensure bash is explicitly called
SHELL ["/bin/bash", "-c"]

# Install the dependencies listed in apt-packages.txt
COPY apt-packages.txt /tmp/apt-packages.txt
RUN apt-get update && \
    xargs apt-get install -y < /tmp/apt-packages.txt && \
    rm -rf /var/lib/apt/lists/*
  
# Instalar Firefox desde el canal oficial
RUN wget -O /tmp/firefox.tar.xz "https://download.mozilla.org/?product=firefox-latest&os=linux64&lang=en-US" && \
    mkdir -p /opt/firefox && \
    tar -xJf /tmp/firefox.tar.xz -C /opt/firefox --strip-components=1 && \
    ln -sf /opt/firefox/firefox /usr/bin/firefox && \
    echo 'export PATH=/opt/firefox:$PATH' >> /etc/profile.d/firefox.sh
     
# Add commands to configure dbus and the GNOME environment   
RUN mkdir -p /var/run/dbus && \
    dbus-uuidgen > /var/lib/dbus/machine-id

# Instalar Python 3.11.2 desde código fuente
#RUN wget https://www.python.org/ftp/python/3.11.2/Python-3.11.2.tgz && \
#    tar xvf Python-3.11.2.tgz && \
#    cd Python-3.11.2 && \
#    ./configure --enable-optimizations && \
#    make -j$(nproc) && \
#    make altinstall && \
#    ln -sf /usr/local/bin/python3.11 /usr/bin/python3 && \
#    ln -sf /usr/local/bin/pip3.11 /usr/bin/pip3

# Install Python 3.11 precompiled
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install --reinstall -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# virtual environment on /opt/venv using Python 3.11
RUN python3 -m venv /opt/venv

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
#ADD https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh /opt/ibgateway-installer.sh
#RUN chmod +x /opt/ibgateway-installer.sh

# Copy entrypoint script and make it executable
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/bin/bash", "/usr/local/bin/entrypoint.sh"]

# Keep the container alive by running bash as the final command
CMD ["bash"]