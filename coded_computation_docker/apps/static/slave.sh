appPath=$APP_PATH
echo $appPath
masterOrSlave=$M_OR_S
echo $masterOrSlave
slaveNum=$SLAVE_NUM

#taskset -c 0 python -u SlaveServer.py $slaveNum N
taskset -c 0 python -u poly_SlaveServer.py $slaveNum N
