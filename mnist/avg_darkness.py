"""
Classify digits by average darkness, as a crude baseline.
"""

from collections import Counter
import mnist_data

plotting = 1
if plotting: import matplotlib.pyplot as plt

all_digits = range(10)

def baseline():
    mn = mnist_data.MNIST()

    sums = [0] * 10
    counts = [0] * 10
    # TODO can we vectorize this?
    for img, label in mn.training.pairs():
        sums[label] += darkness(img)
        counts[label] += 1
    avgs = [sums[s] / (counts[s] or 1) for s in all_digits]
    print(avgs)
    
    def classify(img):
        d = darkness(img)
        def cost(label): return abs(d - avgs[label])
        return min(all_digits, key=cost)
    
    imgs   = mn.test.examples
    labels = mn.test.labels
    
    print(mn.display(imgs[0]))
    print(labels[0], classify(imgs[0]))
    print()

    print(mn.display(imgs[-1]))
    print(labels[-1], classify(imgs[-1]))
    print()

    total = Counter()
    good = Counter()
    for img, label in mn.test.pairs():
        total[label] += 1
        if classify(img) == label:
            good[label] += 1
    ngood = sum(good.values())
    #ngood = sum(classify(img) == label for img, label in mn.test.pairs())
        
    print(f"On the test set the baseline got {ngood} out of {len(imgs)}")
    
    if plotting:
        t = sum(avgs)
        plt.xticks(all_digits)
        plt.plot(all_digits, [avgs[d] / t for d in all_digits], label='avgs')
        plt.plot(all_digits, [good[d] / (total[d] or 1) for d in all_digits], label='good')
        t = sum(total.values())
        plt.plot(all_digits, [total[d] / t for d in all_digits], label='total')
        plt.legend()
        plt.show()

def darkness(img):
    return sum(img)

if __name__ == "__main__":
    baseline()
