import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

np_y = np.array([1,2,3,4,5])
np_predictions = np.array([1,2,4,3,5])
m, b =  np.polyfit(np_y, np_predictions, 1)
plt.figure()
plt.scatter(np_y, np_predictions)
plt.plot(np_y, b + m*np_y)
# plt.title(f"Predicted vs True like:avg, MAPE: {meanAbsPercentDiff}")
plt.ylabel('Predicted like:avg')
plt.xlabel('True like:avg')
for i in range(0,len(np_predictions),2):
    if len(np_y[i:i+2]) > 1:
        print(i)
        print(np_predictions[i:i+2])
        print(np_y[i:i+2])
        print(spearmanr(np_predictions[i:i+2], np_y[i:i+2]).correlation)

x = np.array([1,1,1,1,1,2,2,3,4,4,5,6,7,7, 8, 8, 9, 9, 12, 14, 15, 16, 15, 18])
fig, axes = plt.subplots()
counts, bins, patches = axes.hist(np.array(x), bins=5, edgecolor='black')
axes.set_xticks(bins)
axes.set_xticklabels(bins + [0.1111], rotation=45)
# plt.tight_layout()



axes.set_title('Title')
axes.set_xlabel('Abs diff between prediction and true')
axes.set_ylabel('Count')
plt.show()
# fig.savefig(f"models/concatModel_{dataname}_hist.png")
fig.savefig(f"models/histooooo.png", bbox_inches='tight')
