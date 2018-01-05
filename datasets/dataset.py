import numpy as np


# class to store mnist data
class DataSet(object):
    def __init__(self, images, labels, augment=False):
        # Convert from [0, 255] -> [0.0, 1.0] -> [-1.0, 1.0]
        images = images.astype(np.float32)
        # images = images - 0.5
        # images = 2.0 * images
        self.image_size = 28
        self._num_examples = len(images)
        images = np.reshape(images, (self._num_examples, self.image_size, self.image_size, 1))
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = images[perm]
        self._labels = labels[perm]
        self._augment = augment
        self.pointer = 0
        self.upsize = 1 if self._augment else 0
        self.min_upsize = 2
        self.max_upsize = 2
        self.random_perm_mode = False
        self.num_classes = 10

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size=100, with_label=True, one_hot=False):
        if self.pointer >= self.num_examples - 2 * batch_size:
            self.pointer = 0
        else:
            self.pointer += batch_size
        result = []

        upsize_amount = np.random.randint(self.upsize * self.min_upsize, self.upsize * self.max_upsize + 1)

        # def random_flip(x):
        #  if np.random.rand(1)[0] > 0.5:
        #    return np.fliplr(x)
        #  return x

        def upsize_row_once(img):
            old_size = img.shape[0]
            new_size = old_size + 1
            new_img = np.zeros((new_size, img.shape[1], 1))
            rand_row = np.random.randint(1, old_size - 1)
            new_img[0:rand_row, :] = img[0:rand_row, :]
            new_img[rand_row + 1:, :] = img[rand_row:, :]
            new_img[rand_row, :] = 0.5 * (new_img[rand_row - 1, :] + new_img[rand_row + 1, :])
            return new_img

        def upsize_col_once(img):
            old_size = img.shape[1]
            new_size = old_size + 1
            new_img = np.zeros((img.shape[0], new_size, 1))
            rand_col = np.random.randint(1, old_size - 1)
            new_img[:, 0:rand_col, :] = img[:, 0:rand_col, :]
            new_img[:, rand_col + 1:, :] = img[:, rand_col:, :]
            new_img[:, rand_col, :] = 0.5 * (new_img[:, rand_col - 1, :] + new_img[:, rand_col + 1, :])
            return new_img

        def upsize_me(img, n=self.max_upsize):
            new_img = img
            for i in range(n):
                new_img = upsize_row_once(new_img)
                new_img = upsize_col_once(new_img)
            return new_img

        for data in self._images[self.pointer:self.pointer + batch_size]:
            result.append(self.distort_image(upsize_me(data, upsize_amount), upsize_amount))

        if len(result) != batch_size:
            print "uh oh, self.pointer = ", self.pointer
        assert (len(result) == batch_size)
        result_labels = self.labels[self.pointer:self.pointer + batch_size]
        assert (len(result_labels) == batch_size)
        if one_hot:
            result_labels = np.eye(self.num_classes)[result_labels]
        if with_label:
            return self.scramble_batch(np.array(result, dtype=np.float32)), result_labels
        return self.scramble_batch(np.array(result, dtype=np.float32))

    def distort_batch(self, batch, upsize_amount):
        batch_size = len(batch)
        row_distort = np.random.randint(0, self.image_size + upsize_amount - self.image_size + 1, batch_size)
        col_distort = np.random.randint(0, self.image_size + upsize_amount - self.image_size + 1, batch_size)
        result = np.zeros(shape=(batch_size, self.image_size, self.image_size, 1), dtype=np.float32)
        for i in range(batch_size):
            result[i, :, :, :] = batch[i, row_distort[i]:row_distort[i] + self.image_size,
                                 col_distort[i]:col_distort[i] + self.image_size, :]
        return result

    def scramble_batch(self, batch):
        if self.random_perm_mode:
            batch_size = len(batch)
            result = np.copy(batch)
            result = result.reshape(batch_size, self.image_size * self.image_size)
            result = result[:, self.random_key]
            return result
        else:
            result = batch
            return result

    def distort_image(self, img, upsize_amount):
        row_distort = np.random.randint(0, self.image_size + upsize_amount - self.image_size + 1)
        col_distort = np.random.randint(0, self.image_size + upsize_amount - self.image_size + 1)
        result = np.zeros(shape=(self.image_size, self.image_size, 1), dtype=np.float32)
        result[:, :, :] = img[row_distort:row_distort + self.image_size, col_distort:col_distort + self.image_size, :]
        return result

    def shuffle_data(self):
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]


def read_data_sets(mnist_data):
    class DataSets(object):
        pass

    data_sets = DataSets()

    data_sets.train = DataSet(mnist_data.train.images, mnist_data.train.labels, augment=True)
    data_sets.valid = DataSet(mnist_data.validation.images, mnist_data.validation.labels, augment=False)
    data_sets.test = DataSet(mnist_data.test.images, mnist_data.test.labels, augment=False)
    XDIM = data_sets.train.image_size
    # random_key = np.random.permutation(XDIM*XDIM)
    # data_sets.train.random_key = random_key
    # data_sets.valid.random_key = random_key
    # data_sets.test.random_key = random_key
    return data_sets
