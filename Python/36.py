
import boto3

# AWS Configuration
REGION = "us-east-1"
INSTANCE_ID = "i-xxxxxxxxxxxxxxxxx"

ec2 = boto3.client("ec2", region_name=REGION)

# Check instance health
def check_instance_status():
    response = ec2.describe_instance_status(InstanceIds=[INSTANCE_ID])
    if response["InstanceStatuses"]:
        status = response["InstanceStatuses"][0]["InstanceState"]["Name"]
        return status
    return "unknown"

# Restart instance if it's down
def restart_instance():
    print(f"Restarting instance {INSTANCE_ID}...")
    ec2.start_instances(InstanceIds=[INSTANCE_ID])

if __name__ == "__main__":
    status = check_instance_status()
    print(f"Instance Status: {status}")

    if status in ["stopped", "terminated"]:
        restart_instance()
    else:
        print("Instance is running normally.")
