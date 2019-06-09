import os
import yaml


class Config(object):
    PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def load(cls, file_path):
        stream = open(file_path)
        data = yaml.load(stream)
        for (k, v) in data.items():
            setattr(cls, k, v)
        return cls
