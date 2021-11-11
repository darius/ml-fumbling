"""
Smoke test
"""

import mnist_data

def smoke_test():
    mn = mnist_data.MNIST()

    assert len(mn.training.examples) == 50000
    assert len(mn.test.examples) == 10000

    imgs   = mn.training.examples
    labels = mn.training.labels
    print(mn.display(imgs[0]))
    print(labels[0])
    print()
    print(mn.display(imgs[-1]))
    print(labels[-1])

if __name__ == "__main__":
    smoke_test()
