from flurs.data.entity import User, Item, Event
from flurs.recommender import MFRecommender
import json
import numpy as np
import faust

app = faust.App(
    'flurs-recommender',
    broker='kafka://localhost:9092',
    value_serializer='raw',
)

topic = app.topic('flurs-events', value_type=bytes)

recommender = MFRecommender(k=40)
recommender.initialize()

n_user, n_item = 943, 1682

for u in range(1, n_user + 1):
    recommender.register(User(u - 1))

for i in range(1, n_item + 1):
    recommender.register(Item(i - 1))


@app.agent(topic)
async def process(stream):
    async for obj in stream:
        event = json.loads(obj)
        if event['rating'] < 3:
            continue
        user, item = User(event['user'] - 1), Item(event['item'] - 1)
        print(recommender.recommend(user, np.arange(0, n_item)))
        recommender.update(Event(user, item))
