import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def scaler(df):
    """ Expects a dataframe and normalises all 
        columnsto the 0-1 range. It also returns 
        dataframes with minimum and maximum for
        transforming the cluster centres"""

    # Uses the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df-df_min) / (df_max - df_min)

    return df, df_min, df_max


def backscale(arr, df_min, df_max):
    """ Expects an array of normalised cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr




file_name = 'API_EG.USE.PCAP.KG.OE_DS2_en_csv_v2_5358565.csv'
df = pd.read_csv(file_name, skiprows=4)

columns_to_use = [str(year) for year in range(1971, 2015)]
df_years = df[['Country Name', 'Country Code'] + columns_to_use]

df_years = df_years.fillna(df_years.mean())

df_normalized, df_min, df_max = scaler(df_years[columns_to_use])

inertia = []
num_clusters = range(1, 11)

for k in num_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_normalized)
    inertia.append(kmeans.inertia_)

    
plt.figure(figsize=(12, 8))
plt.plot(num_clusters, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

optimal_clusters = 3 # Choose the appropriate number of clusters

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_years['Cluster'] = kmeans.fit_predict(df_normalized)

cluster_centers = backscale(kmeans.cluster_centers_, df_min, df_max)

plt.figure(figsize=(12, 8))

for i in range(optimal_clusters):
    cluster_data = df_years[df_years['Cluster'] == i][columns_to_use].mean()
    plt.plot(columns_to_use, cluster_data, label=f'Cluster {i+1}', linewidth=2)

plt.plot(columns_to_use, cluster_centers.T, 'k+', markersize=10, label='Cluster Centers')

plt.xlabel('Year')
plt.ylabel('Energy use (kg of oil equivalent per capita)')
plt.title('Energy Use Clustering')

# Adjust x-axis labels to display in intervals of 5 years
years_interval = [str(year) for year in range(1971, 2015, 5)]
plt.xticks(years_interval, years_interval)

plt.legend()
plt.show()


# Code For Comparison Between Clusters

# Define the list of famous and prominent countries
famous_countries = ['United States', 'China', 'Russia', 'Japan', 'Germany', 'Brazil', 'United Kingdom', 'India', 'Canada', 'France', 'Nigeria', 'Switzerland', 'Ukraine', 'Kenya']

# Perform K-means clustering and assign cluster labels
# (Assuming the previous code for clustering is already executed)

# Get the list of unique cluster labels
unique_labels = df_years['Cluster'].unique()

# Set the number of countries to select from each cluster
countries_per_cluster = 10

# Initialize the DataFrame to store the results
result_df = pd.DataFrame()

# Generate the list of countries and mean emissions for each cluster
for label in unique_labels:
    # Filter countries in the current cluster
    cluster_countries = df_years[df_years['Cluster'] == label]
    
    # Select the famous countries from the cluster
    famous_countries_cluster = cluster_countries[cluster_countries['Country Name'].isin(famous_countries)]
    if len(famous_countries_cluster) >= countries_per_cluster:
        countries = famous_countries_cluster['Country Name'].tolist()
    else:
        # If there are not enough famous countries, randomly sample the remaining countries
        remaining_countries = cluster_countries[~cluster_countries['Country Name'].isin(famous_countries)]
        countries = famous_countries_cluster['Country Name'].tolist() + remaining_countries['Country Name'].sample(n=countries_per_cluster - len(famous_countries_cluster), random_state=42).tolist()
    
    # Calculate the mean emissions for each country in the cluster
    emissions_values = cluster_countries.loc[cluster_countries['Country Name'].isin(countries), columns_to_use].mean(axis=1).tolist()
    
    # Create the column names for the cluster
    countries_column = f"Cluster {label} countries"
    emissions_column = f"Cluster {label} mean emissions"
    
    # Add the countries and mean emissions to the result DataFrame
    result_df[countries_column] = countries
    result_df[emissions_column] = emissions_values

# Print the resulting DataFrame
print(result_df)


# A Comparative Bar Plot

import matplotlib.pyplot as plt

# Set the figure size
plt.figure(figsize=(12, 8))

# Iterate over the unique cluster labels
for label in unique_labels:
    # Get the column names for the cluster
    countries_column = f"Cluster {label} countries"
    emissions_column = f"Cluster {label} mean emissions"
    
    # Get the countries and mean emissions for the cluster
    countries = result_df[countries_column]
    mean_emissions = result_df[emissions_column]
    
    # Create the bar plot for the cluster
    plt.bar(countries, mean_emissions, label=f"Cluster {label}")

# Set the x-axis tick labels vertically
plt.xticks(rotation=60)

# Add labels and title to the plot
plt.xlabel("Countries")
plt.ylabel("Mean Emissions")
plt.title("Comparison of Mean Emissions Across Clusters")

# Add a legend
plt.legend()

# Display the plot
plt.show()


# Visually Apealing Dataframe.

# Apply background gradient color to the DataFrame
styled_df = result_df.style.background_gradient(cmap='Greens')

# Display the styled DataFrame
styled_df


from scipy.optimize import curve_fit
from scipy.stats.distributions import  t

# Define the functions
def f(x, a, b, c):
    return a * x**2 + b * x + c

# incoorporating the err_ranges function
def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper   


file_name = 'API_EG.USE.PCAP.KG.OE_DS2_en_csv_v2_5358565.csv'
df = pd.read_csv(file_name, skiprows=4)
columns_to_use = [str(year) for year in range(1971, 2015)]
df_years = df[['Country Name', 'Country Code'] + columns_to_use]
df_years = df_years.fillna(df_years.mean())
country_name = 'Germany'
data = df_years[df_years['Country Name'] == country_name][columns_to_use].values.flatten()
x = np.arange(1971, 2015)
popt, pcov = curve_fit(f, x, data)

x_full_range = np.arange(1971, 2035)
y_full_range = f(x_full_range, *popt)

# Calculate confidence intervals
alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
n = len(x)    # number of data points
p = len(popt) # number of parameters
dof = max(0, n - p) # number of degrees of freedom

# student-t value for the dof and confidence level
tval = t.ppf(1.0-alpha/2., dof) 

# Get standard deviations of the parameters
p_sigmas = np.sqrt(np.diag(pcov))

# Calculate standard deviation of the predictions
y_err = np.sqrt(sum((tval * p_sigma * np.gradient(y_full_range, param))**2 for p_sigma, param in zip(p_sigmas, popt)))

# Confidence intervals
y_lower = y_full_range - y_err
y_upper = y_full_range + y_err

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, data, 'bo', label='Data')
ax.plot(x_full_range, y_full_range, 'r-', label='Best-fitting function')

# Plot the confidence intervals
ax.fill_between(x_full_range, y_lower, y_upper, color='gray', alpha=0.5, label='Confidence range')

ax.set_xlabel('Year')
ax.set_ylabel('Energy use (kg of oil equivalent per capita)')
ax.set_title(f'{country_name} Energy Use')
ax.legend()

plt.show()


