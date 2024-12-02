#!/bin/bash

# Start X virtual framebuffer
#Xvfb :99 -screen 0 1024x768x16 &

# Set DISPLAY environment variable for headless mode
#export DISPLAY=:99

#echo "Generating locales..."
#locale-gen en_US.UTF-8
#update-locale LANG=en_US.UTF-8

# Ensure the correct permissions on the PostgreSQL data directory
chown -R postgres:postgres /var/lib/postgresql/14/main
chown -R postgres:postgres /home/dolph_user/*sql
chown -R dolph_user:dolph_user /home/dolph_user/data

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


# Starting IB Gateway in API mode (headless)
echo "Starting IB Gateway in API mode.."
# /opt/ibgateway-installer.sh -q -dir /root/Jts -overwrite | tee /tmp/ibgateway_install_log.txt
#xvfb-run -a /root/Jts/ibgateway -g -t -ibcApiOnly


# Allow all IPs on port 443
#iptables -A INPUT -p tcp -s 0.0.0.0/0 --dport 443 -j ACCEPT

# Drop all other incoming connections
#iptables -A INPUT -j DROP

#echo "iptables rules configured."

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

# Ensure cron is running and crontab is configured
echo "Configuring crontab..."
cat <<EOF | crontab -
59 23 * * * sudo -u dolph_user /home/dolph_user/stop.sh >> /var/log/stop_cron_test.log 2>&1
0 0 * * * sleep 1 && sudo -u dolph_user /home/dolph_user/compress_logs.sh >> /var/log/compress_logs.log 2>&1
0 0 * * * sleep 10 && sudo -u postgres /usr/lib/postgresql/14/bin/pg_ctl -D /var/lib/postgresql/14/main -l /var/log/db_restart_cron_test.log restart >> /var/log/db_restart_cron_test.log 2>&1
0 0 * * * sleep 20 && sudo -u dolph_user /home/dolph_user/start.sh >> /var/log/start_cron_test.log 2>&1
EOF

echo "Starting cron service..."
service cron start


# Environment setup
export USER=dolph_user
export HOME=/home/dolph_user
export LANG=en_US.UTF-8
export LANGUAGE=en_US:en
export LC_ALL=en_US.UTF-8
export XDG_RUNTIME_DIR=/tmp/runtime-$USER

echo "Setting up runtime directory for $USER..."
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR
chown -R $USER:$USER $XDG_RUNTIME_DIR

# Set up .vnc directory
echo "Setting up VNC directory..."
mkdir -p $HOME/.vnc
chmod 700 $HOME/.vnc
chown -R $USER:$USER $HOME/.vnc

# Load VNC password
if [ -f /home/dolph_user/vnc.sql ]; then
    echo "Loading VNC password from /home/dolph_user/vnc.sql..."
    VNC_PASSWORD=$(grep -oP '(?<=VNC_PASSWORD=")[^"]+' /home/dolph_user/vnc.sql)
    if [ -z "$VNC_PASSWORD" ]; then
        echo "Error: VNC password is empty or not found in vnc.sql!"
        exit 1
    fi
else
    echo "Error: vnc.sql file not found!"
    exit 1
fi

# Create VNC password file if it doesn't exist
if [ ! -f $HOME/.vnc/passwd ]; then
    echo "Creating VNC password file..."
    echo "$VNC_PASSWORD" | su -c "vncpasswd -f > $HOME/.vnc/passwd" $USER
    chmod 600 $HOME/.vnc/passwd
    chown $USER:$USER $HOME/.vnc/passwd
else
    echo "VNC password file already exists."
fi

# Check and clean stale lock files
echo "Cleaning up stale lock files..."
if [ -f /tmp/.X1-lock ]; then
    echo "Removing stale lock file: /tmp/.X1-lock"
    rm -f /tmp/.X1-lock
fi
if [ -d /tmp/.X11-unix ]; then
    echo "Removing stale socket files in /tmp/.X11-unix"
    rm -f /tmp/.X11-unix/X1
    chmod 1777 /tmp/.X11-unix
fi

# Create the xstartup file for Xfce
echo "Creating Xfce startup script..."
cat <<EOF > $HOME/.vnc/xstartup
#!/bin/bash
xrdb $HOME/.Xresources
startxfce4 &
EOF
chmod +x $HOME/.vnc/xstartup
chown $USER:$USER $HOME/.vnc/xstartup

# Start the VNC server
echo "Starting VNC server..."
su -c "vncserver :1 -geometry 1920x1080 -depth 24" $USER
if [ $? -ne 0 ]; then
    echo "Error: Failed to start VNC server!"
    exit 1
fi

# Force-start Xfce
echo "Starting Xfce session..."
DISPLAY=:1 su -c "startxfce4 &" $USER

# Start the DBus daemon
echo "Starting D-Bus service..."
mkdir -p /var/run/dbus
chmod 755 /var/run/dbus
dbus-daemon --system --fork

echo "VNC server and Xfce session are running on DISPLAY=:1."



echo "Starting Dolph..."
sudo -u dolph_user /home/dolph_user/deploy.sh 1

# Keep the container running (in Kubernetes)
tail -f /var/log/postgresql/postgresql-14-main.log


# Return control to the bash shell
# exec "$@"
