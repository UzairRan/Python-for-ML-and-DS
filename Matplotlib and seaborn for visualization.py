#!/usr/bin/env python
# coding: utf-8

# Introduction to Visualization Libraries

# 1- Plots using Matplotlib
# 
# 2- Plots using Seaborn
# 
# Import the required libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# to suppress warnings 
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# import the 'tips' dataset from seaborn
tips_data = sns.load_dataset('tips')

# display head() of dataset
tips_data.head()


# 1- Plots using Matplotlib

# 1.1- Line Plot
# 
# A line graph is the simplest plot that displays the relationship between one independent variable and one dependent dataset. In this plot the points are joined by straight line segments

# In[3]:


# import data
import numpy as np
X = np.linspace(1,20,100)
Y = np.exp(X)

# line plot
plt.plot(X, Y)

# display the plot
plt.show()


# The above line can not be represented by solid line only, but also a dotted line with varied thickness. The points can be marked explicitly using any symbol

# In[4]:


# import data
import numpy as np
X = np.linspace(1,20,100)
Y = np.exp(X)

# line plot
# the argument 'r*' plots each point as red '*'
plt.plot(X, Y, 'r*')

# display the plot
plt.show()


# There can be multiple line plots in one plot. Let's plot three plots to gether in a single graph. Also, add a plot title

# In[5]:


# data
X = np.linspace(1,20,100)
Y1 = X
Y2 = np.square(X)
Y3 = np.sqrt(X)

# line plot
plt.plot(X, Y1, 'r', X, Y2, 'b', X,Y3, 'g')

# add title to the plot
plt.title('Line Plot')

# display the plot
plt.show()


# 1.2- Scatter plot
# 
# A scatter plot is a set of points plotted on horizontal and vertical axes. The scatter plot can be used to study the correlation between two variables. One can also detect the extreme data points using scatter plot.

# In[6]:


# check the head() of tips dataset
tips_data.head()


# In[7]:


# plot the scatter plot for the variable 'total_bill' and 'tip'
# data
X = tips_data['total_bill']
Y = tips_data['tip']

# plot the scatter plot
plt.scatter(X,Y)

# add the axes label to the plot
plt.xlabel('total_bill')
plt.ylabel('tip')

# display the plot
plt.show()


# We can add different colors, opacity and shape of data points. Let's add these customizations in the above plot.

# In[8]:


# plot the scatter plot for the variable 'total_bill' and 'tip'

X = tips_data['total_bill']
Y = tips_data['tip']

# plot the scatter plot
# s is for shape, c is for colour, alpha is for opacity(0 < alpha < 1)
plt.scatter(X,Y, s = np.array(Y)**2, c= 'green', alpha = 0.8)

# add title
plt.title('Scatter Plot')

# add the axes label to the plot
plt.xlabel('total_bill')
plt.ylabel('tip')

#  display the plot
plt.show()


# The bubbles with greater radius display that the tip amount is as compared to the bubbles with less radius

# 1.3- Bar Plot
# 
# A bar plot is used to display categorical data with bars lengths proportional to the values that they represent. The comparison between different categories of categorical variable can be done by studying a bar plot. 
# Im the vertical bar plot, The X_axis displays the categorical variable and Y_axis contain the values corresponding to different categories.

# In[9]:


# check the head() of tips dataset.
tips_data.head()


# In[10]:


# The variable 'smoker' is categorical
# check categories in the variable
set(tips_data['smoker'])


# In[11]:


# bar plot to get the count of smokers and non_smokers in the data
# kind = 'bar' plots the bar plot
# 'rot' = 0 returns the categorical labels horizontally
tips_data.smoker.value_counts().plot(kind = 'bar', rot = 0)

# display the plot
plt.show()


# In[13]:


# bar plot to get the count of smokers and non_smokers in the data
# kind = 'bar' plots the bar plot
# 'rot' = 0 returns the categorical labels horizontally
# color can be used to add a specific colour
tips_data.smoker.value_counts().plot(kind = 'bar', rot = 0, color = 'green')

# plt.text() add the text to the plot 
# x and y are the positions to the axes
# s is the text to added
plt.text(x = -0.05, y = tips_data.smoker.value_counts()[1]+1, s = tips_data.smoker.value_counts()[1])
plt.text(x = 0.98, y = tips_data.smoker.value_counts()[0], s = tips_data.smoker.value_counts()[0])

# add title and axes labels
plt.title('barplot')
plt.xlabel('smoker')
plt.ylabel('Count')

# display the plot
plt.show()


# 1.4- Pie Plot
# 
# Pie plot is a graphical representation of univariate data. It is a circular graph divided intpo slices displaying the numerical proportion. For the categorical variable, each slice of the plot corresponds to each of the categories.

# In[14]:


# Check the head() of dataset
tips_data.head()


# In[16]:


# categories in the day variable
tips_data.day.value_counts(normalize = True)


# In[19]:


# Plot the occurrence of different  days in the dataset
#'autopct' displays the percentage upto 1 decimal place
# 'radius' set the radius of pie plot
plt.pie(tips_data.day.value_counts(), autopct = '%.1f%%', radius = 1.2, labels = ['Sat', 'Sun', 'Thur', 'Fri'])

# display the plot
plt.show()


# Exploded pie plot: 
# 
# is a plot in which one or more sectors are separated from disc

# In[20]:


# Plot the occurrence of different  days in the dataset

plt.pie(tips_data.day.value_counts(), autopct = '%.1f%%', radius = 1.2, labels = ['Sat', 'Sun', 'Thur', 'Fri'], 
       explode = [0,0,0,0.5])
# display the plot
plt.show()


# Donut Pie Plot:
# 
# is a type of pie plot in which there is a hollow centre repreesenting a doughnut

# In[27]:


# Plot the occurrence of different  days in the dataset

# pie plot
plt.pie(tips_data.day.value_counts(), autopct = '%.1f%%', radius = 1.2, labels = ['Sat', 'Sun', 'Thur', 'Fri'])

# add a circle to the centre 
circle  = plt.Circle((0,0), 0.5, color = 'white') 
plot = plt.gcf()
plot.gca().add_artist(circle)

# Display the plot
plt.show()


# 1.5- Histogram
# 
# A histogram is used tpo display the distribution and spread of the continuous variable. One axis represent the range of variable and the other axis shows the frequency of vthe data points. In a histogram there are no gaps between the bars.

# In[28]:


# check the head() of 'tips' dataset
tips_data.head()


# In a tips dataset tip is a continuous variable. Let's plot the histogram to understand the distribution of variable.

# In[30]:


# plot the histogram
# specify the number of bins using 'bins' parameter
plt.hist(tips_data['tip'], bins = 5)

#add the graph title and axes labels
plt.title('Distribution of tip amount')
plt.xlabel('tip')
plt.ylabel('Frequency')

# Display the plot
plt.show()


# From the above plot we can see that tip amount is positively skewed

# 1.6- Box Plot
# 
# Box Plot is a way to visualize the five-number summary of the variable. The five number summary includes the numerical quantities like minimum, first Quartile(Q1), Median(Q2), third Quartile(Q3) and maximum. Box Plot gives information about the outliers in the data. Detecting and removing outliers is one of the most important step in exploratory data analysis. Box Plot also tells about the distribution of data.

# In[31]:


# check the head() of tips dataset
tips_data.head()


# In[33]:


# Plot the distribution of the total_bill
plt.boxplot(tips_data['total_bill'])

# add labels for five number summary
plt.text(x = 1.1, y =tips_data['total_bill'].min(), s = 'min')
plt.text(x = 1.1, y = tips_data.total_bill.quantile(0.25), s= 'Q1')
plt.text(x= 1.1, y = tips_data['total_bill'].median(), s = 'median (Q2)')
plt.text(x = 1.1, y= tips_data.total_bill.quantile(0.75), s = 'Q3')
plt.text(x= 1.1, y = tips_data['total_bill'].max(), s = 'max')

# add the graph title and axes labels
plt.title("Box Plot of total_bill amount")
plt.ylabel("Total_bill")

# display the plot
plt.show()


# The above boxplot clearly shows the presence of outliers above the horizontal line. We can add an arrow to showcase the outliers. Also, the median(Q2) is represented by the orange line which is near the Q1 rather than Q3. This shows that the totall_bill is positively skewed.

# In[34]:


# Plot the distribution of the total_bill
plt.boxplot(tips_data['total_bill'])

# add labels for five number summary
plt.text(x = 1.1, y =tips_data['total_bill'].min(), s = 'min')
plt.text(x = 1.1, y = tips_data.total_bill.quantile(0.25), s= 'Q1')
plt.text(x= 1.1, y = tips_data['total_bill'].median(), s = 'median (Q2)')
plt.text(x = 1.1, y= tips_data.total_bill.quantile(0.75), s = 'Q3')
plt.text(x= 1.1, y = tips_data['total_bill'].max(), s = 'max')

# add an arrow(annotate) to show the Outliers
plt.annotate('Outliers', xy = (0.97,45), xytext = (0.7, 44),
            arrowprops = dict(facecolor = 'black', arrowstyle = 'simple'))
# add the graph title and axes labels
plt.title("Box Plot of total_bill amount")
plt.ylabel("Total_bill")

# display the plot
plt.show()


# 2- Plots using Seaborn
# 
# Seaborn is a python visualization library based on matplotlib. The library provides a high level interface for plotting statistical graphics. As the libraray uses matplotlib in the beckand. We can use the functions in matplotlib along with functions in seaborn. Various functions in the seaborn library allows us to plot complex and advanced statistical plots like linear/higher_order regression, univariate/multivariate distribution, violin , swarm, strip plot, correlation and so on.

# 2.1 Strip Plot
# 
# Strip plot resembles a scatter plot when one variable is categorical. This plot can help study the underlying functions

# In[36]:


import seaborn as sns


# In[37]:


# check the head() of tips data set
tips_data.head()


# In[38]:


# Plot a strip plot  to check the relationship between variables 'tip' and 'time'.
sns.stripplot(y = 'tip', x = 'time', data = tips_data)
# display the plot
plt.show()


# In[39]:


# plot a strip plot with jitter to spread the points

sns.stripplot(y = 'tip', x = 'time', data = tips_data, jitter = True)
# display the plot
plt.show()


# 2.2 Swarm Plot
# 
# Swarm plot is similar to strip plot but it avoids the overlapping of points. This can give better representation of distribution of data.

# In[40]:


# check the head() of tips dataset
tips_data.head()


# In[43]:


# plot the swarm plot for variable time and tip.
sns.swarmplot(y = 'tip', x = 'time', data= tips_data)

# display the plot
plt.show()


# The above plot gives a good representation of tip amount for the time. It can be seen that tip amount is 2 for most of observations. We can see that swarm plot gives a better understanding of variable than strip plot.
# We can add another categorical variable in the above plot by using parameter 'hue

# In[44]:


# Swarm plot with one more categorical variable 'day'
sns.swarmplot(y = 'tip', x = 'time', data= tips_data, hue = 'day')
# display the plot
plt.show()


# The plot shows that tip was collected at lunch time only on Thursday and Friday. The amount of tip collected at dinner time on Saturday is highest.

# 
# 
# 2.3 Violin Plot
# 
# Violin Plot is similar to Box Plot that features a kernel density estimation of the underlying distribution. The plot shows the distribution of numerical variables across categories of one (or more) categorical variables such that those distributions can be compared 

# In[45]:


# Check the head() of tips dataset
tips_data.head()


# In[49]:


# Draw a Violin plot for numerical variable 'total_bill' and categorical variable 'day'
sns.violinplot(y = 'total_bill', x = 'day', data = tips_data)
# display the plot
plt.show()


# In[50]:


# setthe figure size
plt.figure(figsize= (8,5))

# Violin plot with addition of variable 'sex'
# 'split = True' draws half plot for each of category of 'sex'
sns.violinplot(y = 'total_bill', x = 'day', data = tips_data, hue = 'sex',split = True)

# display the plot
plt.show()


# There is no significant difference in the distribution of bill amount and sex

# 2.4 Pair Plot
# 
# A pair plot gives the pair wise distribution of variables in the dataset. Pairplot() function creates a matrix such that each grid shows the relationship between pair of variables. On the diagonal axes a plot shows the univariate distribution of each variable.

# In[51]:


# check the head() of dataset
tips_data.head()


# In[52]:


# plot a pair plot for the tips dataset
# set the figuresize 
plt.figure(figsize = (8,8))

# plot a pair plot
sns.pairplot(tips_data)
# display the plot
plt.show()


# The above plot shows the relationship between all the numerical variables. 'total_bill' and 'tip' has a  positive linear relationship with each other. Also, 'total_bill' and 'tip' are positively skewed. 'Size' has significant impact on 'total_bill', as the minimum bill amount is increasing  with an increasing number of customers(size).
# 

# 
# 2.5 Distribution plot
# 
# A seaborn provides a distplot() function which is used to visualize the distribution of univariate variable. This function uses matplotlib to plot histogram and fit a kernel density estimate(KDE).

# In[53]:


# check the head()of the tips dataset
tips_data.head()


# In[54]:


# Plot a distribution plot of 'total_bill'
sns.distplot(tips_data['total_bill'])

# dsplay the plot
plt.show()


# In[55]:


# Iterate the distplot() function over the time
# list of time
time = ['Lunch', 'Dinner']

# iterate through time 
for i in time:
    subset = tips_data[tips_data['time'] == i]
# Draw the density plot
# 'hist = False' will not plot a histogram
# 'kde = True' plots density curve
    sns.distplot(subset['total_bill'], hist = False, kde = True,
                kde_kws = {'shade': True},
                label = i)


# It can be seen that distribution plot for lunch is more right_skewed than a plot for dinner. This implies that a customer is spending a more time on dinner than lunch   

# 
# 2.6 Count Plot
# 
# Count plot shows the count of each observation in each category of a categorical variable. We can add another variable by using a parameter 'hue'

# In[56]:


# Plot the count of 0observations for each day based on 'time'
# set 'time' as hue parameter
sns.countplot(data = tips_data, x = 'day', hue = 'time')

# display the plot
plt.show()


# 2.7 Heatmap 
# 
# Heatmap is a 2_dimensional graphical representation of data where the individual values that are contained in a matrix are represented as color. Each square in heatmap shows the correlation between variable on each axis.
# Correlation is a statistic that  measure a degree to which two variables move with each other.

# In[57]:


# check the head() of tips dataset
tips_data.head()


# In[58]:


# compute correlation
corr_matrix = tips_data.corr()
corr_matrix  
# Remember that correlation is calculated only  between numerical variables


# In[59]:


# plot heatmap
# 'annot = True' returns the correlation values
sns.heatmap(corr_matrix, annot = True)

# display the plot
plt.show()


# In[ ]:




