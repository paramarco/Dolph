# README

# To build the Image just once
sudo docker build -t dolph-container .
    
# every time the server has to be started    
sudo docker run -it --name dolph-container \
    --entrypoint /usr/local/bin/entrypoint.sh \
    -v ~/data-dolph-container:/home/dolph_user/data \
    -v ~/pgdata:/var/lib/postgresql/14/main \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -p 5432:5432 \
    dolph-container /bin/bash

# To Start a New Shell as root
sudo docker exec -it --user root dolph-container /bin/bash

# To Start a New Shell as dolph_user
sudo docker exec -it --user dolph_user dolph-container /bin/bash

# To display images currently installed 
 sudo docker images
 
# To remove images currently installed 
 sudo docker rmi

# To display on-going running containers
 sudo docker ps -a
 
# To remove any on-going containers
 sudo docker rm
  
# Manual steps to be done just once in the container to check the persistency

root@container:/#  cp /home/dolph_user/*sql /var/lib/postgresql/

(venv) root@container:/# su - postgres
 
postgres@container:~$ vi create_user_db.sql

postgres@container:~$ psql -U postgres -f create_user_db.sql

(venv) root@container:/# su - postgres

postgres@container:~$ vi create_tables.sql

postgres@container:~$ psql -U postgres -d dolph_db -f create_tables.sql 

##Steps to Export/Import the Table Security into PostgreSQL,in development container

root@container:/#   su - postgres

postgres@container:~$ pg_dump -U dolph_user -d dolph_db -t security --data-only -f dump.sql


##Switch to the postgres user in the Kubernetes container
(since the PostgreSQL process runs as the postgres user):

root@container:/#  su - postgres

postgres@container:~$ psql -U postgres -d dolph_db -f dump.sql

# Manual step to access the data

sudo docker exec -it --user root dolph-container /bin/bash

$ cp /var/lib/postgresql/14/main/pg_hba.conf /etc/postgresql/14/main/pg_hba.conf

$ sudo -u postgres /usr/lib/postgresql/14/bin/pg_ctl -D /var/lib/postgresql/14/main -l /var/log/postgresql/postgresql-14-main.log restart

# Dry-run 

(py37) root@dolph-deployment-5b4d776645-6thmq:/# su - dolph_user

$ bash

dolph_user@dolph-deployment-5b4d776645-6thmq:~$ . /opt/venv/bin/activate

(venv) dolph_user@dolph-deployment-5b4d776645-6thmq:~$ python /home/dolph_user/Dolph/DolphRobot.py


#Deployment steps
Kubernetes Registry

Generating new identification details
Please note down the information you need to log in to the Harbor user interface.

$ sudo docker images
$ sudo docker login 8xv7t7tg.c1.de1.container-registry.ovh.net
$ sudo docker tag dolph-container:latest 8xv7t7tg.c1.de1.container-registry.ovh.net/e1256adf-9c74-4b6c-a238-86bbfc8fe1f9/dolph-container:latest
$ sudo docker push 8xv7t7tg.c1.de1.container-registry.ovh.net/e1256adf-9c74-4b6c-a238-86bbfc8fe1f9/dolph-container:latest

#Update Kubernetes Deployment to Pull the Image

In your deployment.yml, make sure you update the image section to reference the image in your OVH registry:


#Create the Kubernetes Secret to Access the Private OVH Registry

Install kubectl on, Download the Latest Version of kubectl

$ curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

Make kubectl Executable, After downloading the kubectl binary, make it executable:

$ chmod +x kubectl

Move the Binary to Your PATH

$ sudo mv kubectl /usr/local/bin/

Verify

$ kubectl version --client

Point kubectl to the kubeconfig file:

$ export KUBECONFIG=/home/afrodita/Dolph/Configuration/kubeconfig.yml

Check if kubectl is correctly configured to communicate with your OVH Kubernetes cluster:

$ kubectl cluster-info

$ echo "export KUBECONFIG=~/Dolph/Configuration/kubeconfig.yml" >> ~/.bashrc

$ source ~/.bashrc

$ kubectl create secret docker-registry ovh-registry-secret \
  --docker-server=8xv7t7tg.c1.de1.container-registry.ovh.net \
  --docker-username=<Loggin_Kubernetes_Registry_OVH.sql> \
  --docker-password=<Loggin_Kubernetes_Registry_OVH.sql> \
  --docker-email=<Loggin_Kubernetes_Registry_OVH.sql> 

#Create Persistent Volume (PV) and Persistent Volume Claim (PVC)
    To persist the data for the directories /home/dolph_user/data and 
    /var/lib/postgresql/14/main, you will create two PVCs.
    YAML for PVC (pvc.yml). Now, apply the PVCs using kubectl:
    
    $ kubectl apply -f pvc.yml
    
    This will create the persistent volume claims for /home/dolph_user/data and 
    /var/lib/postgresql/14/main.
    
    $ kubectl get pvc

#Apply the Deployment

Now, apply the deployment YAML file:

$ kubectl apply -f deployment.yml

    ##Check the PVCs:
    
    $ kubectl get pvc
    
    ##Check the pods:
    
    $ kubectl get pods
    
    ##Check the services:
    
    $ kubectl get svc


##If the pods are not running or there are any issues, you can inspect the details of the pods by running
    $ kubectl describe pod <your-pod-name>
    
    $ kubectl apply -f service.yml
    
    $ kubectl logs -f <your-pod-name>

#Rebuild the Docker Image

    Once you modify the entrypoint.sh file, you’ll need to rebuild the Docker image
     and push it to your OVH registry:
     # Rebuild the image
    $ sudo docker build -t dolph-container:latest .

# Tag the image for the OVH registry
$ sudo docker tag dolph-container:latest 8xv7t7tg.c1.de1.container-registry.ovh.net/e1256adf-9c74-4b6c-a238-86bbfc8fe1f9/dolph-container:latest
# Push the image to the OVH registry
$ sudo docker push 8xv7t7tg.c1.de1.container-registry.ovh.net/e1256adf-9c74-4b6c-a238-86bbfc8fe1f9/dolph-container:latest


# Redeploy the Pod in Kubernetes
    Once the image has been updated in the registry, you can redeploy the pod by 
    deleting the existing pod (Kubernetes will automatically recreate it):
    
    $ kubectl delete pod <your-pod-name>
    $ kubectl get pods -w

# To manually access the container 
     And perform the necessary steps from within the container's console, 
     you can use the kubectl exec command. This allows you 
     to execute commands inside the running container
     
     afrodita@afrodita:~/Dolph/Configuration$ kubectl get pods
    NAME                                READY   STATUS    RESTARTS   AGE
    dolph-deployment-5b4d776645-t8d6z   1/1     Running   0          12m
    
    To get an interactive shell inside the container, run:
    $ kubectl exec -it <pod-name> -- /bin/bash
    
    $ kubectl exec -it <pod-name> --user=dolph_user -- /bin/bash

#Find out which node your pod is running on, following command:

kubectl get pod <pod-name> -o wide

# To install pgAdmin on your client

To install pgAdmin on your laptop (running Debian 11) and connect it to your PostgreSQL server, follow these steps:

##Step 1: Add the pgAdmin Repository

    You need to add the pgAdmin repository to your system’s list of repositories. Download the key:

        curl -fsSL https://www.pgadmin.org/static/packages_pgadmin_org.pub -o pgadmin_key.asc
    
    Move the key to the /etc/apt/trusted.gpg.d/ directory:
      
        sudo mv pgadmin_key.asc /etc/apt/trusted.gpg.d/pgadmin_key.asc
    
    Verify the Repository. First, make sure that the correct repository is added:
    
        echo "deb https://ftp.postgresql.org/pub/pgadmin/pgadmin4/apt/$(lsb_release -cs) pgadmin4 main" | sudo tee /etc/apt/sources.list.d/pgadmin4.list
   
    Update the package list to ensure the newly added repository is recognized:
    
        sudo apt update


##Step 2: Install pgAdmin

    Update the package list and install pgAdmin 4:
    
        sudo apt update
        sudo apt install pgadmin4
    
    During the installation, pgAdmin will be configured to run in desktop mode.

##Step 3: Start pgAdmin

    Locate the pgadmin4 Binary
        sudo find / -name pgadmin4
        example : 
       afrodita@afrodita:~/Dolph/Configuration$ export PATH=$PATH:/usr/pgadmin4/bin/pgadmin4
        afrodita@afrodita:~/Dolph/Configuration$ source ~/.bashrc

    You can start pgAdmin from your desktop environment or by running:
    
        pgadmin4

##Step 4: Connect pgAdmin to Your PostgreSQL Server

    Open pgAdmin 4.

    Click on the “Add New Server” button or right-click on "Servers" and select "Create" -> "Server".

    Fill in the connection details:
        Name: Any name you prefer (e.g., "My PostgreSQL Server").
        Host name/address: The IP address or hostname of the server running PostgreSQL (e.g., localhost if it's the same machine, or the IP of your Docker container).
        Port: Default PostgreSQL port 5432.
        Maintenance database: Typically postgres.
        Username: The PostgreSQL user you want to connect with (e.g., dolph_user).
        Password: The password for the specified user.

    Save the server connection.

##Step 5: Verify the Connection

    Once connected, you should see the databases on your server listed under the server in pgAdmin. You can now manage and query your PostgreSQL databases using pgAdmin.
    
    This setup allows you to manage your PostgreSQL server running in the Docker container from the pgAdmin application installed on your laptop.
      
     sudo apt-get update && sudo apt-get install -y net-tools

#Development procedures

    source Dolph/myenv/bin/activate
    
    spyder3 &

    sudo docker ps -a
    
    sudo docker rm <ID>
    
    sudo docker run -it --name dolph-container --entrypoint /usr/local/bin/entrypoint.sh \
    -v ~/data-dolph-container:/home/dolph_user/data -v ~/pgdata:/var/lib/postgresql/14/main \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -p 5432:5432  dolph-container /bin/bash

    git commit -am "Dolph can alles"

    cat ~/*tok*
    
    git push origin main


    kubectl exec -it dolph-deployment-5b4d776645-6thmq -- /bin/bash
    
    /usr/pgadmin4/bin/pgadmin4 &