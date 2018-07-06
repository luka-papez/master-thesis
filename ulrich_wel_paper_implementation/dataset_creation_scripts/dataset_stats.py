import os
from collections import Counter
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

stats_file = './dataset_stats.pickle'

freqs = None
freqs_48 = None
if not os.path.exists(stats_file):
    freqs = Counter()
    for root_dir, dirs, files in os.walk('../dataset_labels'):
        for f in tqdm(files):
            freqs.update((open(os.path.join(root_dir, f), 'r').read().split(' ')))

    freqs = freqs.most_common()
    pickle.dump(freqs, open(stats_file, 'wb'))
else:
    freqs = pickle.load(open(stats_file, 'rb'))
    freqs_48 = pickle.load(open('./dataset_stats_48.pickle', 'rb'))

print(freqs)
freqs = freqs[::-1]
x_ticks = [k for k, v in freqs]
x = range(len(x_ticks))
y = [v for k, v in freqs]

plt.plot(x, y)
plt.xticks(x, x_ticks, rotation=-90)
plt.yticks(rotation=-90)

freqs_48 = freqs_48[::-1]
x_ticks = [k for k, v in freqs_48]
x = [v - 17 for v in range(len(x_ticks))]
y = [v for k, v in freqs_48]

plt.plot(x, y)
plt.xticks(x, x_ticks, rotation=-90)
plt.yticks(rotation=-90)

plt.show()
