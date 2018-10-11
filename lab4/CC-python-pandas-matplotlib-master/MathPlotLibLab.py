#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df= pd.read_csv("scottish_hills.csv")
print(df.head(10))


# In[3]:


sorted_hills = df.sort_values(by = ['Height'], ascending = False)
print(sorted_hills.head(5))


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[6]:


x = df.Height
y = df.Latitude
z = df.Longitude
plt.scatter(x,y)
plt.savefig("scottish_scatter_plot.png")


# In[7]:


from scipy.stats import linregress
stats = linregress(x,y)


# In[8]:


m = stats.slope


# In[9]:


b=stats.intercept


# In[10]:


plt.scatter(x,y)
plt.plot(x,m*x+b, color="red")


# In[11]:


plt.figure(figsize = (10,10))
plt.scatter(x,y,marker = 'x')
plt.plot(x,m*x+b, color = "red", linewidth=3)
plt.xlabel("Height (m)" , fontsize=20)
plt.ylabel("Latitude", fontsize =20)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)


# In[13]:


import numpy as np
colors = np.random.rand(len(y))
plt.scatter(y,z,s = (x-900), c=colors, alpha =0.5)


# In[14]:


plt.hist(x,bins =25 ,normed =True)
plt.savefig("histogram.png",dpi=25)


# In[17]:


import numpy as np
plt.style.use('seaborn-pastel')
shifted_x = x -100

fig,ax = plt.subplots()
ax.hist(x,bins = 25, normed=True, histtype = 'stepfilled',alpha=0.8, label = 'height')
ax.hist(shifted_x , normed =True , histtype = "stepfilled", alpha = 0.8, label='Height - 100')
ax.legend(prop={'size':10})
ax.set_ylabel('Normalised Distribution')
ax.set_xlabel('Height')


# In[23]:


labels = ['EU','US','AS']
meateaters = [122,135,23]
vegetarians = [40,43,23]

index = np.arange(len(labels))
fig, ax = plt.subplots()
bar_width = 0.35
opacity = 0.6

rectsl = ax.bar(index,meateaters,bar_width,alpha = opacity, color='b',label= 'meateaters')

rects2 = ax.bar(index + bar_width, vegetarians , bar_width, alpha = opacity, color = 'r',label = 'Vegetarians')

ax.set_xlabel('Continent')
ax.set_ylabel('COS emissions')
ax.set_title('CO2 Emissions per continent, per diet')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(labels)
ax.legend()





# In[29]:


def fahrenheit2celsius(temp):
    (5./9.)*(temp -32)
temperature = [100,102,106,105,90,85,89,100,102,103,108,100,102,106,105,90,85,85,89,199,102,103,108]


















# In[31]:


fig, ax_f = plt.subplots()
ax_c = ax_f.twinx()
ax_f.plot(temperature)
ax_f.set_xlim(0,24)

y1,y2 = ax_f.get_ylim()
ax_c.set_ylim(fahrenheit2celsius(y1), fahrenheit2celsius(y2))
ax_c.figure.canvas.draw()

ax_f.set_title('Two scales: Fahrenheit and celsius')
ax_f.set_ylabel('Fahrenheit')
ax_c.set_ylabel('Celsius')
ax_f.set_ylabel('Time (hour of day)')




# In[32]:


newx = np.linspace(0,1,500)
newy = np.sin(4*np.pi*newx)*np.exp(-5 *newx)

fig , ax = plt.subplots()
ax.plot(newx,newy)




# In[33]:


fig, ax =plt.subplots()
ax.fill(newx,newy)


# In[34]:


stockA = [0,10,20,30,40,60,80,140]
plt.plot(stockA)
plt.xlabel('Time')
plt.ylabel("Stock A price")


# In[35]:


plt.boxplot(x)


# In[37]:


plt.figure()
plt.boxplot(x,1)

plt.figure()
plt.boxplot(x , 0,'gD')

plt.figure()
plt.boxplot(x,0,'')

plt.figure()
plt.boxplot(x,0 , 'rs',0)


# In[40]:


df2= pd.read_csv("school_earnings.csv")
print(df2.head(10))


# In[41]:


x = df2.School
y = df2.Women
z = df2.Men
f = df2.Gap


# In[48]:


plt.hist(y,bins =10 ,normed =True)


# In[58]:



fig,ax = plt.subplots()
ax.hist(y,alpha = opacity, label = 'Woman')
ax.hist(z,alpha = opacity, label='Men')
ax.legend(prop={'size':10})
ax.set_ylabel('Salary')
ax.set_xlabel('Density')
ax.set_facecolor((0,0,0))


# In[111]:


fig, ax = plt.subplots()
bar_width = 0.3
index = np.arange(len(x))
bar1 = ax.bar(index, y, bar_width,
                alpha=opacity, color='b',label='Men')
ax.set_xticklabels(x,rotation=90)
bars2 = ax.bar(index + bar_width, z, bar_width,
                alpha=opacity, color='r',label='Women')
ax.set_xticks(index + bar_width/2)
ax.set_xlabel('Schools')
ax.set_ylabel('Salary')
ax.legend()


# In[105]:


fig, ax = plt.subplots()
ax.set_xticklabels(y)
plt.figure()
ax.boxplot([y,z])
ax.set_xticklabels(["woman","men"])
ax.set_xlabel('Gender boxes')
ax.set_ylabel('Salary')


# In[ ]:




