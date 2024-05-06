import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Example data for perfect positive linear relationship
X = np.array([1, 2, 3, 4, 5])
Y = np.array([1, 2, 3, 4, 5])

# Calculate the Pearson correlation coefficient
pearson_corr, _ = pearsonr(X, Y)
print("Pearson correlation coefficient for perfect positive linear relationship:", pearson_corr)

# Plot the data
plt.scatter(X, Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Perfect Positive Linear Relationship')
plt.grid(True)
plt.show()



import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Example data for perfect negative linear relationship
X = np.array([1, 2, 3, 4, 5])
Y = np.array([5, 4, 3, 2, 1])

# Calculate the Pearson correlation coefficient
pearson_corr, _ = pearsonr(X, Y)
print("Pearson correlation coefficient for perfect negative linear relationship:", pearson_corr)

# Plot the data
plt.scatter(X, Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Perfect Negative Linear Relationship')
plt.grid(True)
plt.show()


import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Example data for no linear relationship
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 1, 2, 1, 2])

# Calculate the Pearson correlation coefficient
pearson_corr, _ = pearsonr(X, Y)
print("Pearson correlation coefficient for no linear relationship:", pearson_corr)

# Plot the data
plt.scatter(X, Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('No Linear Relationship')
plt.grid(True)
plt.show()
