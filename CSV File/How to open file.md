## HOW TO OPEN FILE

#  Iris Dataset Analysis

This script loads and displays the **Iris dataset** using `pandas`.

## 🔍 Code

```python
import pandas as pd

# Get the data from the CSV file
a = pd.read_csv("Iris.csv")

# Display the first few rows
print(a)

# Display all rows
for row in a.values:
    print(row)

# Display all column names
print(a.columns)
