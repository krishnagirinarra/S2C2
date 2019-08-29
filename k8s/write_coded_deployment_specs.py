import yaml

template = """
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: {name}
spec:
  template:
    metadata:
      labels:
        app: {label}
    spec:
      nodeSelector:
        kubernetes.io/hostname: {host}
      containers:
      - name: coded-computation
        image: youraccount/yourimage:v1 
        imagePullPolicy: Always 
        ports:
        - containerPort: 57023
        - containerPort: 22
        env:
        - name: APP_PATH 
          value: s2c2 
        - name: APP_NAME 
          value: poly_matMul
        - name: LDPC 
          value: 1
        - name: M_OR_S
          value: {master_or_slave}
        - name: NODE_IPS
          value: {node_ips}
        - name: SLAVE_NUM
          value: {slave_num}
        - name: TOTAL_SLAVES
          value: {total_slaves}
"""


## \brief this function genetares the service description yaml for a task 
# \param kwargs             list of key value pair. 
# In this case, call argument should be, 
# name = {taskname}, dir = '{}', host = {hostname}

def write_profiler_specs(**kwargs):
    specific_yaml = template.format(**kwargs)
    dep = yaml.load(specific_yaml, Loader=yaml.BaseLoader)
    return dep
