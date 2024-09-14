#!/bin/bash

# Ensure the correct permissions on the PostgreSQL data directory
chown -R postgres:postgres /var/lib/postgresql/14/main
chown -R postgres:postgres /home/dolph_user/*sql

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
fi

# Remove stale PID file if it exists
if [ -f /var/lib/postgresql/14/main/postmaster.pid ]; then
    echo "Removing stale PID file..."
    rm /var/lib/postgresql/14/main/postmaster.pid
fi

# Start PostgreSQL
echo "Starting PostgreSQL..."
sudo -u postgres /usr/lib/postgresql/14/bin/pg_ctl -D /var/lib/postgresql/14/main -l /var/log/postgresql/postgresql-14-main.log start

# Wait for PostgreSQL to start
echo "Waiting for PostgreSQL to start..."
while ! pg_isready -q -d postgres://localhost:5432; do
    sleep 1
done
echo "PostgreSQL started."

# Activate the virtual environment
if [ -f /opt/venv/bin/activate ]; then
    . /opt/venv/bin/activate
else
    echo "Virtual environment not found. Skipping activation."
fi

# Keep the container running (in Kubernetes)
tail -f /var/log/postgresql/postgresql-14-main.log


# Return control to the bash shell
# exec "$@"
