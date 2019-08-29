#!/bin/bash
service ssh start

appPath=$APP_PATH
echo $appPath
masterOrSlave=$M_OR_S
echo $masterOrSlave
slaveNum=$SLAVE_NUM
echo $slaveNum
cd $appPath
python updateConfig.py
echo $PWD
#Create a softlink
ln -s /home/zhifeng/apps/s2c2/data /home/zhifeng/apps/static/data
bash -c "ssh-keygen -t rsa -f ~/.ssh/id_rsa -N ''"
sleep 120
IFS=':' read -ra ADDR <<< "$NODE_IPS"
if [ "$masterOrSlave" == "master" ]; then
    for (( i = 1; i < $TOTAL_SLAVES; i++ ))
    do
    bash -c "cat ~/.ssh/id_rsa.pub | sshpass -p 'zhifeng' ssh -p 5000 ${ADDR[$i]} -o StrictHostKeyChecking=no 'mkdir -p ~/.ssh; cat - >> ~/.ssh/authorized_keys'";
    done
  ./master.sh > master.log 2>master.error
  #./master.sh
else
    bash -c "cat ~/.ssh/id_rsa.pub | sshpass -p 'zhifeng' ssh -p 5000 ${ADDR[0]} -o StrictHostKeyChecking=no 'mkdir -p ~/.ssh; cat - >> ~/.ssh/authorized_keys'"
    bash randomInterfere.sh &
    ./slave.sh > slave.log 2>slave.error
  #./slave.sh
fi
while true; do sleep 200; done
