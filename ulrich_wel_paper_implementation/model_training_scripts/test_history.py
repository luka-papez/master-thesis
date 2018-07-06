import os
import pickle
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt

history_folder = '../history'
correspondences = np.array([pickle.load(open(os.path.join(history_folder, f), 'rb'))[0][1] for f in natsorted(os.listdir(history_folder)) if 'correspondence' in f])
edit_distances  = np.array([pickle.load(open(os.path.join(history_folder, f), 'rb'))[0][1] for f in natsorted(os.listdir(history_folder)) if 'edit_distance' in f])

print(correspondences)
accuracies = correspondences.mean(axis=1) / 128.0 * 100
distances = edit_distances.mean(axis=1)

index = [int(x + 1) for x in range(len(distances))]

fig, ax1 = plt.subplots()
ax1.plot(index, accuracies, 'red')
ax1.set_xlabel('iteration')
ax1.set_ylabel('%')

ax2 = ax1.twinx()
ax2.plot(index, distances, 'black')
ax2.set_xlabel('iteration')
ax2.set_ylabel('edits')

fig.tight_layout()

ax1.legend(['accuracy'])
ax2.legend(['edit_distance'])
plt.show()
