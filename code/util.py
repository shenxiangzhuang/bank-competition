import time


class Timeit:
    def __init__(self, name='Process'):
        self.start = None
        self.end = None
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        print(f'{self.name} Spends {self.end - self.start} s')


if __name__ == '__main__':
    with Timeit('NaiveBayes[Train]') as t:
        time.sleep(3)
        print('training model...')
