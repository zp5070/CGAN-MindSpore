import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as CT
from mindspore.common import dtype as mstype
from mindspore import Tensor
def create_dataset(data_path,
                   flatten_size,
                   batch_size,
                   repeat_size=1,
                   num_parallel_workers=1):
    mnist_ds = ds.MnistDataset(data_path)
    type_cast_op = CT.TypeCast(mstype.float32)
    onehot_op = CT.OneHot(num_classes=10)

    mnist_ds = mnist_ds.map(input_columns="label",
                            operations=onehot_op,
                            num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="label",
                            operations=type_cast_op,
                            num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image",
                            operations=lambda x:
                            ((x - 127.5) / 127.5).astype("float32"),
                            num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image",
                            operations=lambda x: (x.reshape((flatten_size, ))),
                            num_parallel_workers=num_parallel_workers)
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


def one_hot(num_classes=10, arr=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    return np.eye(num_classes)[arr]