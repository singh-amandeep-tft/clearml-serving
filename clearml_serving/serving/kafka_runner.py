# kafka_runner.py
import json
import os

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from clearml_serving.serving.main import process_with_exceptions

HOST = os.environ.get("KAFKA_HOST", "localhost:9092")
GROUP = os.environ.get("KAFKA_GROUP", "model_service_group")
INPUT_TOPIC = os.environ.get("KAFKA_INPUT_TOPIC", "model_input")
OUTPUT_TOPIC = os.environ.get("KAFKA_OUTPUT_TOPIC", "model_output")


async def start_kafka_loop():
    consumer = AIOKafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=HOST,
        group_id=GROUP,
        enable_auto_commit=True,
    )

    producer = AIOKafkaProducer(
        bootstrap_servers=HOST
    )

    await consumer.start()
    await producer.start()

    try:
        print(f"Started consumer on topic: {INPUT_TOPIC}")
        async for msg in consumer:
            try:
                payload = json.loads(msg.value)

                model_id = payload["model_id"]
                request_data = payload["data"]
                
                # Process request
                return_value = await process_with_exceptions(
                    base_url=model_id,
                    version=None,
                    request=request_data,
                    serve_type="process",
                )

                # Publish the result to output topic
                result_msg = json.dumps({
                    "model_id": model_id,
                    "result": return_value,
                }).encode("utf-8")

                await producer.send_and_wait(OUTPUT_TOPIC, result_msg)
                print(f"Published result to {OUTPUT_TOPIC}")

            except Exception as e:
                print(f"Error processing Kafka message: {e}")
    finally:
        print("Stopping consumer and producer")
        await consumer.stop()
        await producer.stop()
