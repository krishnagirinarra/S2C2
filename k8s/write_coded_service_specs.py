import yaml

template = """
apiVersion: v1
kind: Service
metadata:
  name: {name}
  labels:
    purpose: coded-computation
spec:
  ports:
  - port: 5000
    targetPort: 22
    name: scp-port 
  - port: 57023
    targetPort: 57023
    name: python-port 
  selector:
    app: {name}
"""

## \brief this function genetares the service description yaml for a task 
# \param kwargs             list of key value pair. 
# In this case, call argument should be, name = {taskname}
def write_profiler_service_specs(**kwargs):
    # insert your values
    specific_yaml = template.format(**kwargs)
    dep = yaml.load(specific_yaml)
    return dep
