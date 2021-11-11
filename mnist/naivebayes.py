"""
Classify digits by Naive Bayes, as a baseline.
"""

import numpy as np
import mnist_data
from math import log

all_digits = np.arange(10)

def is_dark(pixel):
    return pixel < 50    # TODO or what?

def baseline():
    mn = mnist_data.MNIST()
    training = mn.training

    # TODO vectorize?
    counts = [[[0, 0] for i in range(len(training.examples[0]))]
              for d in all_digits]
    for img, label in training.pairs():
        count_by_pixel = counts[label]
        for i, pixel in enumerate(img):
            dark = is_dark(pixel)
            count_by_pixel[i][dark] += 1
    evidences = [[(log((nlight + 1) / total),         # add-1 smoothing
                   log((ndark + 1)  / total),)
                  for [nlight, ndark] in count_by_pixel
                  for total in [nlight + ndark + 2]]
                 for count_by_pixel in counts]

    def classify(img):
        # TODO vectorize?
        logp = [sum(ev[is_dark(pixel)]
                    for ev, pixel in zip(evidence, img))
                for evidence in evidences]
        return max(all_digits, key=lambda d: logp[d]) # sheesh
    
    ngood = sum(classify(img) == label for img, label in mn.test.pairs())
    #ngood = 0
        
    print(f"On the test set, naivebayes")
    print(f"  got {ngood} out of {len(mn.test)}")
        
if __name__ == "__main__":
    baseline()
