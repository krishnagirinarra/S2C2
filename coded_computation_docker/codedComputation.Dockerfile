# start from base
FROM tensorflow/tensorflow:latest

# install system-wide deps for python and node
RUN apt-get -yqq update
RUN apt-get install -yqq openssh-client openssh-server bzip2 wget net-tools sshpass parallel 

RUN echo 'root:zhifeng' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#MaxStartups 10:30:60/MaxStartups 100/' /etc/ssh/sshd_config

# fetch app specific deps
RUN ["pip", "install", "keras"]

#RUN apt-get -yqq install python-pip python-dev
RUN mkdir /home/zhifeng

# copy our application code
ADD apps /home/zhifeng/apps
WORKDIR /home/zhifeng/apps

# expose port
EXPOSE 22 57023

# start app
CMD [ "bash", "./startApp.sh" ]
