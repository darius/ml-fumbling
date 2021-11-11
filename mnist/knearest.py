"""
Classify digits by k-nearest-neighbors, as a baseline.
"""

from collections import Counter
import numpy
from numpy.random import default_rng
import heapq
import mnist_data

rng = default_rng()

K = 1                           # Number of nearest neighbors to use
sample_size = 1000              # Too slow with more than 1000

def distance(img1, img2):
    return ((img1 - img2) ** 2).sum()

def mode(xs):
    return Counter(xs).most_common(1)[0][0]

def baseline():
    mn = mnist_data.MNIST()
    training = mn.training.shuffled(rng).slice(0, sample_size)
    
    def classify(img):
        def distance_to_img(i):
            return distance(training.examples[i], img)
        knn = heapq.nsmallest(K, range(len(training)), key=distance_to_img)
        labels = [training.labels[i] for i in knn]    # TODO use numpy
        if 0:
            print(f'labels: {labels}')
            print(f'mode: {mode(labels)}')
        return mode(labels)
    
    ngood = sum(classify(img) == label for img, label in mn.test.pairs())
        
    print(f"On the test set, {K}-nearest-neighbors (sampling {sample_size})")
    print(f"  got {ngood} out of {len(mn.test)}")
        
if __name__ == "__main__":
    baseline()
