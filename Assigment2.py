import pandas as pd # import panda library as pd for data manipulation
import matplotlib.pyplot as plt# import matplotlib as plt for data visualitzation 
from matplotlib import style
from scipy import stats # import scipy as stats for statistic calculation
import numpy as np # import nump as np 
import seaborn as sns

def read_worldbank(filename: str):
    # Read the file into a pandas dataframe
    dataframe = pd.read_csv(filename)
    
    # Transpose the dataframe
    df_transposed = dataframe.transpose()
    
    # Populate the header of the transposed dataframe with the header information 
   
    # silice the dataframe to get the year as columns
    df_transposed.columns = df_transposed.iloc[1]
    # As year is now columns so we don't need it as rows
    df_transposed_year = df_transposed[0:].drop('year')
    
    # silice the dataframe to get the country as columns
    df_transposed.columns = df_transposed.iloc[0]
    
    # As country is now columns so we don't need it as rows
    df_transposed_country = df_transposed[0:].drop('country')
    
    return dataframe, df_transposed_country, df_transposed_year

# Passing filename to real_worldbank function 
original_df, df_by_country, df_by_year = read_worldbank('wb_climatechange.csv')

# show the first 10 row
original_df.head()

# show the first 10 row
df_by_country.head()

# show the first 10 row
df_by_year.head()

# we want to see countries power comsumption over specfic years
# we need to filter our original data frame to get specific fields
data = original_df[['country','year','electric_power_comsumption']]

# drop the null values present in the dataset
data= data.dropna()


year_1990 = data[data['year'] == 1990] # filter data related to 1990 
year_1995 = data[data['year'] == 1995] # filter data related to 1995 
year_2000 = data[data['year'] == 2000] # filter data related to 2000 
year_2005 = data[data['year'] == 2005] # filter data related to 2005 
year_2010 = data[data['year'] == 2010] # filter data related to 2010 

x = np.arange(11) # make 11 groups for 11 countries 

def plot_ep_barplot():
    style.use('ggplot')

    # set fig size
    plt.figure(figsize=(15,10))

    # set width of bars
    barWidth = 0.1 

    # plot bar charts
    plt.bar(x,year_1990['electric_power_comsumption'],color='lime', width=barWidth, label='1990')
    plt.bar(x+0.2,year_1995['electric_power_comsumption'],color='red',width=barWidth, label='1995')
    plt.bar(x+0.4,year_2000['electric_power_comsumption'],color='blue',width=barWidth, label='2000')
    plt.bar(x+0.6,year_2005['electric_power_comsumption'],color='darkblue',width=barWidth, label='2005')
    plt.bar(x+0.8,year_2010['electric_power_comsumption'],color='darkred',width=barWidth, label='2010')

    # show the legends on the plot
    plt.legend()

    # set the x-axis label
    plt.xlabel('Country',fontsize=8)

    # add title to the plot 
    plt.title("Electric Power Comsumption per capita",fontsize=10)

    # add countries names to the 11 groups on the x-axis
    plt.xticks(x+0.2,('Afghanistan', 'Australia', 'Belgium', 'Brazil', 'Canada', 'China',
               'India', 'Iran, Islamic Rep.', 'Pakistan', 'United States','UK'),fontsize=8,rotation = 45)

    # show the plot
    plt.show()

plot_ep_barplot()

# we want to see countries population growth over the years
# we need to filter our original data frame to get specific fields
population_growth_data = original_df[['country','year','population_growth']]

# drop the null values present in the dataset
population_growth_data= population_growth_data.dropna()

population_growth_year_1990 = population_growth_data[population_growth_data['year'] == 1990]  # filter data related to 1990 
population_growth_year_1995 =population_growth_data[population_growth_data['year'] == 1996]  # filter data related to 1995 
population_growth_year_2000 = population_growth_data[population_growth_data['year'] == 2000]  # filter data related to 2000 
population_growth_year_2005 = population_growth_data[population_growth_data['year'] == 2005]  # filter data related to 2005 
population_growth_year_2010 = population_growth_data[population_growth_data['year'] == 2010]  # filter data related to 2010 

x_group = np.arange(12) # make 12 groups for 12 countries 

# defining a function which will plot bar chart
# of Population growth
def plot_pop_barplot():
    style.use('ggplot')

    # set fig size
    plt.figure(figsize=(15,10))

    # set width of bars
    barWidth = 0.1 

    # plot bar charts
    plt.bar(x_group,population_growth_year_1990['population_growth'],color='lime', width=barWidth, label='1990')
    plt.bar(x_group+0.2,population_growth_year_1995['population_growth'],color='red',width=barWidth, label='1995')
    plt.bar(x_group+0.4,population_growth_year_2000['population_growth'],color='blue',width=barWidth, label='2000')
    plt.bar(x_group+0.6,population_growth_year_2005['population_growth'],color='darkblue',width=barWidth, label='2005')
    plt.bar(x_group+0.8,population_growth_year_2010['population_growth'],color='darkred',width=barWidth, label='2010')

    # show the legends on the plot
    plt.legend()

    # set the x-axis label
    plt.xlabel('Country',fontsize=10)

    # add title to the plot 
    plt.title("Population Growth ",fontsize=10)

    # add countries names to the 11 groups on the x-axis
    plt.xticks(x_group+0.2,('Afghanistan', 'Australia', 'Belgium', 'Brazil', 'Canada', 'China',
               'India', 'Iran, Islamic Rep.', 'Pakistan', 'United States','UK','Iraq'),fontsize=10,rotation = 45)

    # show the plot
    plt.show()

plot_pop_barplot()

mean = original_df[['country','year','electric_power_comsumption']]
m = mean[mean['country']=='China']
m['electric_power_comsumption']

#calculate mean and median
mean = original_df['electric_power_comsumption'].mean()
median = original_df['electric_power_comsumption'].median()
print(f'The mean ECP is: {mean:.2f}')
print(f'The median ECP is: {median:.2f}')

#calculate mode
mode = stats.mode(original_df['electric_power_comsumption'])
print(f'The mode of ECP is: {mode[0][0]:.2f}')

#calculate standard deviation
std = original_df['electric_power_comsumption'].std()
print(f'The standard deviation of ECP is: {std:.2f}')

#calculate interquartile range
q75, q25 = np.percentile(original_df['electric_power_comsumption'], [75, 25])
iqr = q75 - q25
print(f'The interquartile range of ECP is: {iqr:.2f}')

#calculate skewness
skew = original_df['electric_power_comsumption'].skew()
print(f'The skewness of ECP is: {skew:.4f}')

original_df.describe()

#Assigning the values of specific countries for line chart
india=original_df[original_df['country']== 'India'] 
pakistan=original_df[original_df['country']== 'Pakistan']
united_states=original_df[original_df['country']== 'United States']
canada=original_df[original_df['country']== 'Canada']
belgium=original_df[original_df['country']== 'Belgium']
brazil=original_df[original_df['country']== 'Brazil']
iran=original_df[original_df['country']== 'Iran']
china=original_df[original_df['country']== 'China']
australia=original_df[original_df['country']== 'Australia']
united_kingdom=original_df[original_df['country']== 'United Kingdom']
def plot_urbpop_line():
    #use the filtered data for line chart to compare the data
    plt.plot(india.year, india.urban_population, label="India", linestyle="--" ) 
    plt.plot(pakistan.year, pakistan.urban_population, label="Pakistan", linestyle="--" )
    plt.plot(australia.year, australia.urban_population, label="Australia", linestyle="--" )
    plt.plot(united_states.year, united_states.urban_population, label="US", linestyle="--" )
    plt.plot(united_kingdom.year, united_kingdom.urban_population, label="UK", linestyle="--" )
    plt.plot(brazil.year, brazil.urban_population, label="Brazil", linestyle="--" )
    plt.plot(belgium.year, belgium.urban_population, label="Belgium", linestyle="--" )
    plt.plot(iran.year, iran.urban_population, label="Iran", linestyle="--" )
    plt.plot(canada.year, canada.urban_population, label="Canada", linestyle="--" )
    plt.plot(china.year, china.urban_population, label="China", linestyle="--" )
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

    #set the lagend outside the boundries of line chart for visibility of line chart
    plt.show()
    
# ploting urban population line plot
plot_urbpop_line()

def plot_pc_line():
    plt.plot(india.year, india.electric_power_comsumption, label="India", linestyle="--" )
    plt.plot(pakistan.year, pakistan.electric_power_comsumption, label="Pakistan", linestyle="--" )
    plt.plot(australia.year, australia.electric_power_comsumption, label="Australia", linestyle="--" )
    plt.plot(united_states.year, united_states.electric_power_comsumption, label="US", linestyle="--" )
    plt.plot(united_kingdom.year, united_kingdom.electric_power_comsumption, label="UK", linestyle="--" )
    plt.plot(brazil.year, brazil.electric_power_comsumption, label="Brazil", linestyle="--" )
    plt.plot(belgium.year, belgium.electric_power_comsumption, label="Belgium", linestyle="--" )
    plt.plot(iran.year, iran.electric_power_comsumption, label="Iran", linestyle="--" )
    plt.plot(canada.year, canada.electric_power_comsumption, label="Canada", linestyle="--" )
    plt.plot(china.year, china.electric_power_comsumption, label="China", linestyle="--" )
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')  
    #set the lagend outside the boundries of line chart for visibility of line chart
    plt.show()
    
# ploting line plot for power compsumption
plot_pc_line()

# we want to see countries population growth over the years
# we need to filter our original data frame to get specific fields
agricultural_land_data = original_df[['country','year','agricultural_land']]

# drop the null values present in the dataset
agricultural_land_data= agricultural_land_data.dropna()

# we want to see countries population growth over the years
# we need to filter our original data frame to get specific fields
urban_population_data = original_df[['country','year','urban_population']]

# drop the null values present in the dataset
urban_population_data= urban_population_data.dropna()

corr1_data=pd.merge(data, population_growth_data,  on=['country','year']) #to filter out the null values without loss of data for every feature sepratly
corr2_data=pd.merge(corr1_data, agricultural_land_data,  on=['country','year'])
corr_data=pd.merge(corr2_data, urban_population_data,  on=['country','year'])
corr_data=corr_data.drop(['year'], axis=1)
corr_metrix=corr_data.corr() #finding the correlation metrix

sns.heatmap(corr_metrix, cmap="YlGnBu", annot=True) #draw heat map using seaborn
plt.show()
sns.heatmap(corr_data[corr_data['country'] == 'China'].corr(), cmap="YlGnBu", annot=True).set(
    title="China")
plt.show()
sns.heatmap(corr_data[corr_data['country'] == 'Pakistan'].corr(), cmap="YlGnBu", annot=True).set(
    title="Pakistan") #heatmap for china for analysis
plt.show()
sns.heatmap(corr_data[corr_data['country'] == 'United States'].corr(), cmap="YlGnBu", annot=True).set(
    title="United States") #heatmap for united states

plt.show()
