
import paho.mqtt.client as mqtt
import random
import time
import json

# MQTT Configuration
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "iot/sensor/data"

# Simulate sensor data
def generate_sensor_data():
    return {
        "temperature": round(random.uniform(20.0, 30.0), 2),
        "humidity": round(random.uniform(40.0, 60.0), 2),
        "pressure": round(random.uniform(990.0, 1020.0), 2),
        "timestamp": time.time()
    }

# MQTT Client
client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}")

client.on_connect = on_connect
client.connect(BROKER, PORT, 60)
client.loop_start()

try:
    while True:
        data = generate_sensor_data()
        payload = json.dumps(data)
        client.publish(TOPIC, payload)
        print(f"Published: {payload}")
        time.sleep(2)
except KeyboardInterrupt:
    print("Stopping IoT simulator.")
    client.loop_stop()
    client.disconnect()
