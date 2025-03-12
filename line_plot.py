import pandas as pd

# Test shapes
shapes = ['airplane', 'chair', 'table']

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv("numbers/all2.csv")


import matplotlib.pyplot as plt

threshold = data['threshold'].astype(float).to_list()[:11]
marker = ['o', "v", "^", "<", ">", "*", "+"]
for column, ma in zip(data.columns[1:], marker):
    plt.plot(threshold[:len(data[column].to_list())], data[column].to_list(), label=column, marker=ma)


# Customizing the plot
#plt.title("Performance of Different Methods by Threshold")
plt.xlabel("Geodesic Distance Threshold", fontsize=18)
plt.ylabel("Relative IoU", fontsize=18)
plt.legend(title="Methods", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig('ablation2.pdf')