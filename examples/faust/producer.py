import sys
import json
from kafka import KafkaProducer
from kafka.errors import KafkaError

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda m: json.dumps(m).encode('ascii'))
topic = 'flurs-events'

keys = ['user', 'item', 'rating', 'timestamp']

with open(sys.argv[1], 'r') as f:  # /path/to/ml-100k/u.data
    for line in f.readlines():
        event = dict(zip(keys, map(int, line.rstrip().split('\t'))))

        future = producer.send(topic, event)
        try:
            future.get(timeout=10)
        except KafkaError as e:
            print(e)
            break
