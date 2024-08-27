#!/bin/bash

# Ensure the correct permissions on the PostgreSQL data directory
chown -R postgres:postgres /var/lib/postgresql/14/main
chown -R postgres:postgres /home/dolph_user/*sql

# Copy the custom pg_hba.conf file to the correct location if it exists
echo "Remember to change the authentication method manually";

if [ -f /home/dolph_user/pg_hba.conf ]; then
    cp /home/dolph_user/pg_hba.conf /etc/postgresql/14/main/pg_hba.conf
    chown postgres:postgres /etc/postgresql/14/main/pg_hba.conf
    
    cp /home/dolph_user/postgresql.conf /etc/postgresql/14/main/postgresql.conf
    chown postgres:postgres /etc/postgresql/14/main/postgresql.conf

fi

# Initialize the database if necessary
if [ ! -f /var/lib/postgresql/14/main/PG_VERSION ]; then
    echo "Initializing PostgreSQL data directory..."
    sudo -u postgres /usr/lib/postgresql/14/bin/initdb -D /var/lib/postgresql/14/main
else
    echo "PostgreSQL data directory already exists."
fi

# Remove stale PID file if it exists
if [ -f /var/lib/postgresql/14/main/postmaster.pid ]; then
    echo "Removing stale PID file..."
    rm /var/lib/postgresql/14/main/postmaster.pid
fi

# Start PostgreSQL
echo "Starting PostgreSQL..."
sudo -u postgres /usr/lib/postgresql/14/bin/pg_ctl -D /var/lib/postgresql/14/main -l /var/log/postgresql/postgresql-14-main.log start

# Wait a 30 seconds to ensure PostgreSQL has started
sleep 30

# Activate the virtual environment
source /opt/venv/bin/activate

# Return control to the bash shell
exec "$@"
