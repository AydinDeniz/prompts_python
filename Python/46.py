
from kubernetes import client, config
import time

# Load Kubernetes config
config.load_kube_config()

v1 = client.CoreV1Api()

# Deploy a new pod
def deploy_pod(namespace="default"):
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "my-app"},
        "spec": {
            "containers": [
                {
                    "name": "my-container",
                    "image": "nginx",
                    "ports": [{"containerPort": 80}]
                }
            ]
        }
    }
    
    v1.create_namespaced_pod(namespace=namespace, body=pod_manifest)
    print("Pod 'my-app' deployed successfully.")

# Monitor cluster health
def monitor_cluster():
    while True:
        print("
Checking cluster health...")
        pods = v1.list_pod_for_all_namespaces(watch=False)
        for pod in pods.items:
            print(f"Pod {pod.metadata.name} - Status: {pod.status.phase}")
        time.sleep(10)

if __name__ == "__main__":
    print("Deploying Kubernetes Pod...")
    deploy_pod()

    print("Starting cluster monitoring...")
    monitor_cluster()
