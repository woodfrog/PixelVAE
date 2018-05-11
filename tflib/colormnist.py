import numpy as np

import os
import urllib
import gzip
import pickle
import PIL.Image


def mnist_generator(data, batch_size, n_labelled):
    images = data
    print('total number of images: {}'.format(len(images)))

    images = images.astype('float32')
    if n_labelled is not None:
        labelled = numpy.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)

        if n_labelled is not None:
            np.random.set_state(rng_state)
            np.random.shuffle(labelled)

        total_batches = len(images) / batch_size
        truncted_images = images[:total_batches * batch_size]
        image_batches = truncted_images.reshape(-1, batch_size, 64, 64, 3)
        image_batches = np.transpose(image_batches, axes=[0, 1, 4, 2, 3])

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in xrange(len(image_batches)):
                yield (np.copy(image_batches[i]), np.copy(labelled))
        else:

            for i in xrange(len(image_batches)):
                yield (np.copy(image_batches[i]),)

    return get_epoch


def load(batch_size, test_batch_size, n_labelled=None):
    train_data, dev_data, test_data = read_colormnist_data()

    return (
        mnist_generator(train_data, batch_size, n_labelled),
        mnist_generator(dev_data, test_batch_size, n_labelled),
        mnist_generator(test_data, test_batch_size, n_labelled)
    )


def read_colormnist_data():
    train_dir = '/local-scratch/cjc/GenerativeNeuralModuleNetwork/data/COLORMNIST/TWO_5000_64_modified/train'
    test_dir = '/local-scratch/cjc/GenerativeNeuralModuleNetwork/data/COLORMNIST/TWO_5000_64_modified/test'

    train_images = []
    test_images = []

    for image_file in sorted(os.listdir(train_dir)):
        image_path = os.path.join(train_dir, image_file)
        image = PIL.Image.open(image_path)
        image = np.array(image, dtype=np.float32)
        train_images.append(image)

    train_images = np.stack(train_images, axis=0)

    for image_file in sorted(os.listdir(test_dir)):
        image_path = os.path.join(test_dir, image_file)
        image = PIL.Image.open(image_path)
        image = np.array(image, dtype=np.float32)
        test_images.append(image)

    test_images = np.stack(test_images, axis=0)

    return train_images, test_images, test_images


if __name__ == '__main__':
    train_data, test_data, gen_data = load(16, 16, None)
    train_gen = train_data()

    import IPython;

    IPython.embed()
