# StatsWithSahira

This is a python package to make your lives easier by using python code to do statistical problems. 

## Installation

```
pip install StatsWithSahira
```

## Import 
```python

#You can import the needed functionalities as per your needs

from StatsWithSahira import Mean
from StatsWithSahira import Median
from StatsWithSahira import Mode
from StatsWithSahira import StandardDeviation
from StatsWithSahira import Skewness
from StatsWithSahira import UnivariateRegression
from StatsWithSahira import MultiVariateRegression
```


## Usage 

### Using Mean functionality
```python
# Reading the dataframe For continuous series' 
print('For continuous series'')
mean_df = pd.read_csv('https://raw.githubusercontent.com/sheikh-sahira1703/data/main/mean.csv')
mean_cont = Mean(data=mean_df, x_col='X', f_col='F') # Creating an instance of the mean class
mean_cont.print_mean_from_ci() #Prints the mean for the continuous series

# Reading the dataframe for the discrete series 
print('\n')
print('For discrete series')
discrete_df = pd.read_csv("https://raw.githubusercontent.com/sheikh-sahira1703/data/main/mean_discrete.csv")
mean_discrete = Mean(data=discrete_df, x_col="X", f_col="F") # Creating the instance of the mean class 
mean_discrete.print_mean_discrete() #Printing the mean for the discrete series

#Generating data for the continuous series 
print('\n')
print('For individual series')
series = [i for i in range(1, 51)]
Mean.print_mean_individual(series) # Calling the method to print the mean of individual series
```


### Using the Median functionality
```python
# Reading the dataframe for the continuous series 
print('For continuous series')
median_cont = pd.read_csv("https://raw.githubusercontent.com/sheikh-sahira1703/data/main/median.csv")
median_continuous = Median(data=median_cont, x_col="X", f_col="F") #Creating an instance of the median class
median_continuous.print_df() #Printing the dataframe with the necessary details for finding the median of continuous series
median_continuous.print_median_continuous() #Printing the median of the continuous series with necessary details
median_continuous.print_quartiles() #Printing the quartiles of the continuous series  with the necessary details

#Reading the dataframe for the discrete series
print('\n')
print('For discrete series') 
median_disc = pd.read_csv("https://raw.githubusercontent.com/sheikh-sahira1703/data/main/median_discrete.csv")
median_discrete = Median(data=median_disc, x_col="X", f_col="F") #Creating an instance of the median clas
median_discrete.print_df() #Printing the dataframe with the necessary details for finding the median of discrete series
median_discrete.print_median_discrete() #Printing the median of the discrete series with necessary details
median_discrete.print_quartiles_discrete() #Printing the quartiles of the discrete series with necessary details

#Reading the dataframe for open ended intervals 
print('\n')
print('For open ended series')
df_open_ended = pd.read_csv("https://raw.githubusercontent.com/sheikh-sahira1703/data/main/open_ended.csv")
median_open_ended = Median(data=df_open_ended, x_col="age", f_col="freq") #Creating an instance of the median clas
median_open_ended.print_quartiles_open_ended() #Printing the quartiles of the open ended series series with necessary details

#Initialising the individual series for finding median
print('\n')
print('For individual series')
my_list_median = [i for i in range(1,100)]
Median.print_median_individual(my_list_median) #calculating the median
```

### Using the Mode functionality
```python
#Reading the dataframe for finding the mode of continuous series 
print('For continuous series')
mode_continuous = pd.read_csv("https://raw.githubusercontent.com/sheikh-sahira1703/data/main/mode.csv")
mode_cont = Mode(data=mode_continuous, x_col="X", f_col="F") #Creating instance of Mode class
mode_cont.print_mode_from_ci() #Printing the mode of the continuous series with necessary details 

#Reading the dataframe for finding the mode of discrete series 
print('\n')
print('For discrete series')
mode_disc = pd.read_csv("https://raw.githubusercontent.com/sheikh-sahira1703/data/main/mode_discrete.csv")
mode_discrete = Mode(data=mode_disc,x_col="X", f_col="F") #Creating instance of Mode class
mode_discrete.print_mode_discrete() #Printing the mode of the discrete series with necessary details

#Initialising the individual series for finding mode
print('\n')
print('For individual series')
my_mode = [1,1,5,3,2,9,4,56,4,8,4,5,78,879,1,3,4,5,8,6,4]
Mode.print_mode_individual(my_mode) #calculating the mode
```

### Using the StandardDeviation functionality
```python
#Reading the dataframe for finding the standard deviation of continuous series 
print('For continuous series')
stdev_cont = pd.read_csv("https://raw.githubusercontent.com/sheikh-sahira1703/data/main/stdev_cont.csv")
standardDeviation = StandardDeviation(data=stdev_cont, x_col="X", f_col="F") #Creating instance of StandardDeviation class
standardDeviation.print_stdev_cont() #Printing the standard deviation of the continuous series with necessary details

#Reading the dataframe for finding the standard deviation of discrete series 
print('\n')
print('For discrete series')
stdev_disc = pd.read_csv("https://raw.githubusercontent.com/sheikh-sahira1703/data/main/stdev_discrete.csv")
discreteStdev = StandardDeviation(data=stdev_disc, x_col="X", f_col="F") #Creating instance of StandardDeviation class
discreteStdev.print_stdev_discrete() #Printing the standard deviation of the discrete series with necessary details 

#Generating random data for finding standard deviation of individual series 
print('\n')
print('For individual series')
myList = [5,7,9,3,1,4,6,8,2,10]
individual_stdev = StandardDeviation.print_stdev_individual(myList) #Printing the standard deviation of individual series with necessary details
```

### Using the Skewness functionality
```python
#Reading the dataframe for finding the skewness of continuous series 
print('For continuous series')
skew_continuous = pd.read_csv("https://raw.githubusercontent.com/sheikh-sahira1703/data/main/skew_cont.csv")
skew_cont = Skewness(data=skew_continuous, x_col="X", f_col="F")  #Creating instance of Skewness class
skewness_cont = skew_cont.print_skewness_continuous() #Printing the skewness of the continuous series with necessary details

#Reading the dataframe for finding the skewness of open ended series 
print('\n')
print('For open ended series')
skew_open = pd.read_csv("https://raw.githubusercontent.com/sheikh-sahira1703/data/main/open_ended.csv")
skew_open_ended = Skewness(data=skew_open, x_col="age", f_col="freq") #Creating instance of Skewness class
skewness_open_ended = skew_open_ended.print_skewness_open_ended() #Printing the skewness of the open ended series with necessary details

#Reading the dataframe for finding the skewness of discrete series
print('\n')
print('For discrete series')
skew_disc = pd.read_csv("https://raw.githubusercontent.com/sheikh-sahira1703/data/main/skew_discrete.csv")
skew_discrete = Skewness(data=skew_disc, x_col="X", f_col="F") #Creating instance of Skewness class
skewness_discrete = skew_discrete.print_skewness_discrete() #Printing the skewness of the continuous series with necessary details

#Initialising the series for finding skewness
print('\n')
print('For individual series')
my_skew = [i for i in range(100)]
Skewness.print_skewness_individual(my_skew) #calculating the skewness
```

### Using the UnivariateRegression functionality 

```python
#Reading the dataset for Univariate regression 
data_uni = pd.read_csv("https://raw.githubusercontent.com/sheikh-sahira1703/data/main/Popu-profit.csv")
X = data_uni["col1"].values #storing the x values 
y = data_uni["col2"].values #storing the y values 
reg = UnivariateRegression(X, y) #Creating an instance of UnivariateRegression class 
alpha, beta = reg.calculate_coef() #Calculating the coefficients of the regression
print(f"Alpha is {alpha}\nBeta is {beta}")
eqn = reg.print_equation() #Printing the regression equation 
print(eqn)
pred = reg.predict(X) #Predicting based on our regression 
std_err = reg.standard_error() #Calculating standard error 
print(f"The standard error is {std_err}")
figure = reg.plot_data(title='Regression using own function', xlabel='X', ylabel='Y', figsize=(16, 6)) #Plotting the data
plt.show()
comparison_fig, eqn = reg.compare(figsize=(16, 6)) #Comparing our regression model with StatsModels and SciKitLearn
print(eqn)
plt.show()
r_squared = reg.find_r_squared() #Calculating R squared
print(f"The value of R^2 is {r_squared}")
adjusted_r_squared = reg.find_adjusted_r_squared() #Calculating adjusted R squared 
print(f"The adjusted R^2 is {adjusted_r_squared}")
```

### Using the MultiVariateRegression functionality
```python
#Reading the dataset for Multivariate regression 
data_multi = pd.read_csv("https://raw.githubusercontent.com/sheikh-sahira1703/data/main/multi-regression.csv")
regressor = MultiVariateRegression(data_multi) #Creating an instance of Multivariateregression class
coef = regressor.calculate_coef() #Calculating the coefficients of the regression
print(f"The coefficients are {coef}")
r_squared = regressor.calculate_r_squared() #Calculating R squared
print(f"R squared {r_squared}")
adj_r_sq = regressor.calculate_adjusted_r_squared()
print(f"Adjusted r squared {adj_r_sq}") #Calculating adjusted R squared 
regressor.compare() #Comparing our regression model with StatsModels and SciKitLearn
```


Github profile: [Github profile](https://github.com/sheikh-sahira1703 "Github profile")

Github repository of the package: [Github repository of the package](https://github.com/sheikh-sahira1703/StatsWithSahira "Github repository of the package")

Author email: [Sahira Sheikh](mailto:sheikh_sahira1703@hotmail.com)
