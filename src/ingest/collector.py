import os
import asyncio
import json
from opcua import Client
from kafka import KafkaProducer

# ---------------------------------------------------------------------------
# Environment configuration (passed from docker‑compose)
# ---------------------------------------------------------------------------
OPCUA_ENDPOINT = os.getenv("OPCUA_ENDPOINT", "opc.tcp://plc-refinery:4840")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "refinery.sensors")

# ---------------------------------------------------------------------------
# Tag map – map a friendly name to the OPC‑UA node id. Extend this list with
# all tags you need from your refinery PLCs.
# ---------------------------------------------------------------------------
TAG_MAP = {
    "CDC_TopTemp": "ns=2;i=10",
    "CDC_BotTemp": "ns=2;i=11",
    "Comp_VibrationX": "ns=2;i=20",
    "Comp_VibrationY": "ns=2;i=21",
    "Comp_Temp": "ns=2;i=22",
    # add more tags as needed …
}

async def read_and_publish(client: Client, producer: KafkaProducer):
    """Continuously read the tag list and push a JSON message to Kafka.
    The loop runs at ~10 Hz (adjust `await asyncio.sleep` for your sampling rate)."""
    while True:
        payload = {}
        for name, nodeid in TAG_MAP.items():
            try:
                value = client.get_node(nodeid).get_value()
            except Exception as e:
                # If a tag fails, log a placeholder and continue – we never want the collector to die.
                value = None
            payload[name] = value
        # Encode as UTF‑8 JSON bytes
        producer.send(KAFKA_TOPIC, value=json.dumps(payload).encode())
        await asyncio.sleep(0.1)  # 10 Hz sampling

def main():
    print(f"Connecting to OPC UA endpoint: {OPCUA_ENDPOINT}")
    client = Client(OPCUA_ENDPOINT)
    client.connect()
    print("OPC UA client connected.")

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        compression_type="gzip",
        linger_ms=5,
        max_request_size=5_000_000,
    )
    print(f"Kafka producer ready – sending to {KAFKA_TOPIC}@{KAFKA_BOOTSTRAP}")

    try:
        asyncio.run(read_and_publish(client, producer))
    finally:
        client.disconnect()
        producer.flush()
        producer.close()
        print("Collector shut down gracefully.")

if __name__ == "__main__":
    main()
