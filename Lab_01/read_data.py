import gzip
import os
import struct
import numpy as np
from urllib import request

def download_fashion_mnist(data_dir):
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for file in files:
        unzipped_path = os.path.join(data_dir, file.replace(".gz", ""))
        if not os.path.exists(unzipped_path):
            zipped_path = os.path.join(data_dir, file)
            print(f"Downloading {file}...")
            request.urlretrieve(base_url + file, zipped_path)
            
            print(f"Unzipping {file}...")
            with gzip.open(zipped_path, 'rb') as f_in:
                with open(unzipped_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(zipped_path) 
    print("Dataset is ready.")


def load_fashion_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels