# ##### BAR GRAPH (FREQUENCY) #######

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
# it is used to import the pyplot module from the matplotlib library 
# The pyplot module provides a collection of functions that make it easy to create and customize plots. 

car_data = {'Audi': 124, 'BMW': 98, 'Mercedes': 113}  # The dictionary car_data stores car brands as keys and their frequencies as values
car_data

car_brands = list(car_data.keys())
# This line extracts the keys from the dictionary car_data and converts them into a list. The keys in this case are the car brands (‘Audi’, ‘BMW’, ‘Mercedes’).
frequencies = list(car_data.values())
# This line extracts the values from the dictionary car_data and converts them into a list. The values are the frequencies (124, 98, 113).

## Why We Do This i.e. converting to list?
# >> Compatibility with Plotting Functions: Many plotting functions in libraries like matplotlib expect data to be in list format. By converting the dictionary keys and values into lists, we ensure that the data is in the correct format for plotting.
# >> Separation of Data: Separating the keys and values into two lists makes it easier to manipulate and use them independently in different parts of the code. For example, you can use car_brands for labeling the x-axis and frequencies for the y-axis in a bar chart.

# Plotting Bar Graph

plt.bar(car_brands, frequencies, color=['blue', 'green', 'red'])
plt.xlabel('Car Brands')
plt.ylabel('Frequency')
plt.title('Frequency of Car Brands')
plt.show()

## EXPLANATION OF ABOVE CODE:
# >> plt.bar(): This function creates a bar chart.
# >> car_brands: This is the list of labels for the x-axis (the categories being compared).
# >> frequencies: This is the list of values for the y-axis (the data being plotted).
# >> color: This parameter sets the colors of the bars. Here, ‘blue’, ‘green’, and ‘red’ are assigned to the bars for ‘Audi’, ‘BMW’, and ‘Mercedes’, respectively.
# >> plt.xlabel('Car Brands') : This function sets the label for the x-axis. In this case, it labels the x-axis as “Car Brands”.
# >> plt.ylabel('Frequency') : This function sets the label for the y-axis. Here, it labels the y-axis as “Frequency”.
# >> plt.title('Frequency of Car Brands') : This function sets the title of the chart. The title “Frequency of Car Brands” describes what the chart is about.
# >> plt.show() : Displaying the Chart

# ##### PIE CHART (RELATIVE FREQUENCY) #######

import matplotlib.pyplot as plt

car_data = {'Audi': 124, 'BMW': 98, 'Mercedes': 113}
car_brands = list(car_data.keys())
frequencies = list(car_data.values())

total_frequency = sum(frequencies)
relative_frequencies = [freq / total_frequency for freq in frequencies]

plt.pie(relative_frequencies, labels=car_brands, autopct='%1.1f%%', startangle=140) # We use plt.pie() to create the pie chart.
# The autopct parameter formats the percentages, and startangle rotates the start of the pie chart for better visualization.

plt.title('Relative Frequency of Car Brands') 
plt.show()
