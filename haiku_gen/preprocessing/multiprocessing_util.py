from functools import reduce
from multiprocessing import Pool


class MultiProcessingDatastructure:

    def __init__(self, haiku, pipeline):
        self.haiku = haiku
        self.pipeline = pipeline


def multiprocessing_generator(data, pipeline):
    for d in data:
        yield MultiProcessingDatastructure(d, pipeline)


def apply_pipeline(obj: MultiProcessingDatastructure):
    return reduce(lambda x, f: f(x), obj.pipeline, obj.haiku)


def multiprocess_data(data, pipeline, n_processes=4):
    with Pool(n_processes) as p:
        g = multiprocessing_generator(data, pipeline)
        mapped = p.map(apply_pipeline, g)
    return mapped
