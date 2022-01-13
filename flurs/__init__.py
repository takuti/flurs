from logging import getLogger, StreamHandler, Formatter, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setFormatter(Formatter('%(asctime)s : %(levelname)s : %(process)d : %(message)s'))
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

__version__ = '0.0.4'

__all__ = ['baseline', 'data', 'datasets', 'recommender', 'utils']
