import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (replace 'your_dataset.csv' with your file path)
df = pd.read_csv('your_dataset.csv')

# Ensure the dataset has a time column; replace 'Date' with the name of your date column
df['Date'] = pd.to_datetime(df['Date'])  # Convert to datetime format if necessary
df.set_index('Date', inplace=True)  # Set the date column as the index

# Plotting configuration
plt.figure(figsize=(10, 6))

# Plot each selected column as a line (adjust column names as needed)
# Here, we assume 'Value' represents the data to plot over time
plt.plot(df.index, df['Value'], label='Value Trend', linestyle='-', linewidth=2, marker='o')

# Add titles and labels
plt.title('Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Value')

# Customize aesthetics
plt.grid(True, linestyle='--', alpha=0.5)  # Add grid with style
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(loc='upper left')  # Add a legend in the upper left corner

# Export the plot as a PNG file
plt.tight_layout()  # Adjust layout for better spacing
plt.savefig('trend_plot.png', format='png', dpi=300)  # Save as a high-resolution PNG
plt.show()
