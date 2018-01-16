import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from os import makedirs
from os.path import exists

data_dir = './MNIST_data'
data_file = 'mnist.npz'


class DataFeed(object):
    def __init__(self, shuffle=True):
        self.data = self.fetch_data()
        self.max = self.data['train_images'].shape[0]
        self.val_max = self.data['validation_images'].shape[0]
        self.test_max = self.data['test_images'].shape[0]
        self.mean = self.data['mean']
        self.std = self.data['std']
        self.shuffle = shuffle
        self.permutation = []
        self.offset = 0
        np.random.seed(42)
        if not shuffle:
            # Shuffle this one time and never again
            self.permutation = np.random.permutation(self.max)

        self.reset_permutation()

    def reset_permutation(self):
        if self.shuffle:
            self.permutation = np.random.permutation(self.max)

        self.offset = 0

    def next(self, batch_size):
        start = self.offset
        self.offset = min(self.offset + batch_size, self.max)

        selection = self.permutation[start:self.offset]

        if self.offset == self.max:
            print("resetting")
            self.reset_permutation()

        return self.data['train_images'][selection], self.data['train_labels'][selection]

    def train(self, size=1000):
        selection = np.random.permutation(self.max)[:size]
        return self.data['train_images'][selection], self.data['train_labels'][selection]

    def validation(self, size=1000):
        selection = np.random.permutation(self.val_max)[:size]
        return self.data['validation_images'][selection], self.data['validation_labels'][selection]

    def test(self, rescale=False):
        if rescale:
            return self.data['test_images'] * self.std + self.mean, self.data['test_labels']
        return self.data['test_images'], self.data['test_labels']

    def _normalize(self, data, update=False):
        if update:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
        return div0((data - self.mean), self.std)

    def fetch_data(self):
        if not exists(data_dir):
            makedirs('%s' % data_dir)

        if not exists('%s/%s' % (data_dir, data_file)):
            mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

            train_data = self._normalize(mnist.train.images, True)

            validation_data = self._normalize(mnist.validation.images)
            test_data = self._normalize(mnist.test.images)

            np.savez('%s/%s' % (data_dir, data_file),
                     std=self.std,
                     mean=self.mean,
                     train_images=train_data,
                     train_labels=mnist.train.labels,
                     validation_images=validation_data,
                     validation_labels=mnist.validation.labels,
                     test_images=test_data,
                     test_labels=mnist.test.labels)

            mnist_data = {'mean': self.mean,
                          'std': self.std,
                          'train_images': train_data,
                          'train_labels': mnist.train.labels,
                          'validation_images': validation_data,
                          'validation_labels': mnist.validation.labels,
                          'test_images': test_data,
                          'test_labels': mnist.test.labels}

            # np.save('%s/%s' % (data_dir, data_file),
            #         mnist_data)
        else:
            with np.load('%s/%s' % (data_dir, data_file)) as data:
                # data = np.load('%s/%s' % (data_dir, data_file)).item()

                mnist_data = {
                    'mean': data['mean'],
                    'std': data['std'],
                    'train_images': data['train_images'],
                    'train_labels': data['train_labels'],
                    'validation_images': data['validation_images'],
                    'validation_labels': data['validation_labels'],
                    'test_images': data['test_images'],
                    'test_labels': data['test_labels']
                }

        return mnist_data


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0  # -inf inf NaN
    return c


if __name__ == '__main__':
    # print(fetch_data())
    print("hej")
