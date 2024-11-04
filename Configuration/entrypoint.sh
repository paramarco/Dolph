#!/bin/bash

# Start X virtual framebuffer
Xvfb :1 -screen 0 1024x768x16 &

# Set DISPLAY environment variable for headless mode
export DISPLAY=:1

# Ensure the correct permissions on the PostgreSQL data directory
chown -R postgres:postgres /var/lib/postgresql/14/main
chown -R postgres:postgres /home/dolph_user/*sql

ls -lart /home/dolph_user/

# Check if the PostgreSQL data directory is initialized
if [ "$(ls -A /var/lib/postgresql/14/main)" ]; then
    echo "PostgreSQL data directory already exists."
    
    # Remove any stale files
    if [ -d "/var/lib/postgresql/14/main/lost+found" ]; then
        echo "Removing lost+found directory..."
        rm -rf /var/lib/postgresql/14/main/lost+found
    fi
else
    echo "Initializing PostgreSQL data directory..."
    sudo -u postgres /usr/lib/postgresql/14/bin/initdb -D /var/lib/postgresql/14/main
fi

# Overwrite PostgreSQL config files if they exist
if [ -f /home/dolph_user/pg_hba.conf ]; then
    cp /home/dolph_user/pg_hba.conf /var/lib/postgresql/14/main/pg_hba.conf
    cp /home/dolph_user/postgresql.conf /var/lib/postgresql/14/main/postgresql.conf
    chown postgres:postgres /var/lib/postgresql/14/main/postgresql.conf
    chown postgres:postgres /var/lib/postgresql/14/main/pg_hba.conf
    
    cp /home/dolph_user/*sql /var/lib/postgresql/

else
    echo "there is something wrong there is no config in directory..."    
fi

# Remove stale PID file if it exists
if [ -f /var/lib/postgresql/14/main/postmaster.pid ]; then
    echo "Removing stale PID file..."
    rm /var/lib/postgresql/14/main/postmaster.pid
fi

# Start PostgreSQL
echo "Starting PostgreSQL..."
cp /var/lib/postgresql/14/main/pg_hba.conf /etc/postgresql/14/main/pg_hba.conf
sudo -u postgres /usr/lib/postgresql/14/bin/pg_ctl -D /var/lib/postgresql/14/main -l /var/log/postgresql/postgresql-14-main.log start

# Wait for PostgreSQL to start

echo "Waiting for PostgreSQL to start..."
while ! pg_isready -q -d postgres://localhost:4713; do
    sleep 1
done
echo "PostgreSQL started."


# Activate the virtual environment
if [ -f /opt/venv/bin/activate ]; then
    . /opt/venv/bin/activate
else
    echo "Virtual environment not found. Skipping activation."
fi

# Set up iptables rules to allow only specific connections
echo "Configuring iptables rules..."

# Switch to legacy iptables version
update-alternatives --set iptables /usr/sbin/iptables-legacy

# Allow all IPs on port 443
#iptables -A INPUT -p tcp -s 0.0.0.0/0 --dport 443 -j ACCEPT

# Drop all other incoming connections
#iptables -A INPUT -j DROP

echo "iptables rules configured."

# Install TWS if not already installed
#if [ ! -d "/root/Jts" ]; then
#    echo "Installing Trader Workstation..."
#    /opt/install_tws.expect
#fi

# Start Trader Workstation in the background if installed
#if [ -f "/root/Jts/twsstart.sh" ]; then
#    /root/Jts/twsstart.sh &
#else
#    echo "TWS not installed successfully."
#    exit 1
#fi


# Keep the container running (in Kubernetes)
tail -f /var/log/postgresql/postgresql-14-main.log


# Return control to the bash shell
# exec "$@"
