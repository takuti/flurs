```eh
brew services start zookeeper
brew services start kafka
```

```sh
faust -A recommender worker -l info
```

```sh
python producer.py /path/to/ml-100k/u.data
```
