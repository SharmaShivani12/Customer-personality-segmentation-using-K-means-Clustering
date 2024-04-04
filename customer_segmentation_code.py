#Importing the Libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
from sklearn.compose import ColumnTransformer

data=pd.read_csv('marketing_campaign.csv', sep="\t")
print("Number of datapoints:", len(data))

data.head()
data.info()
#print(f"We have {data.isna().sum()}")

# Check for missing values and sum them up
missing_values = data.isna().sum()

# Print the count of missing values for each column
print(f"missingvalues: \n", missing_values)

# As per the output we have few categorical column need to convert them to numeric as per our requirements:

# converting date column in the date format

data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')
#print(data['Dt_Customer'].head(5))

data["Income"].plot(kind="hist")
plt.show()

################## Feature Engineering ##########################################

# fix missing value in column Income / using group by Education and Marital Status / avg
missing=data.groupby(["Education","Marital_Status"])["Income"].transform("mean").round(0)
data["Income"].fillna(missing, inplace= True)

# Renaming of the columns for better understanding
data=data.rename(columns={'NumWebPurchases': "Web",'NumCatalogPurchases':'Catalog','NumStorePurchases':'Store'})
# Total spending amount
data['Total_Spending']=data['MntWines']+data['MntFruits']+data['MntMeatProducts']+data['MntFishProducts']+data['MntSweetProducts']+data['MntGoldProds']
#print(data['Total_Spending'].head(5))

# Segmentation on the basis of education

data["Education"]=data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

to_drop = [ "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
data = data.drop(to_drop, axis=1)

data.info()

####################### ML Model #########################

#Get list of categorical variables
s = (data.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables in the dataset:", object_cols)

#Label Encoding the object dtypes.
LE=LabelEncoder()
for i in object_cols:
    data[i]=data[[i]].apply(LE.fit_transform)
    
print("All features are now numerical")

# Creating copy of data( for reference purpose and preprocessing)
dc = data.copy()

 #creating a subset of dataframe by dropping the features on deals accepted and promotions
cols_drop = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
dc= dc.drop(cols_drop, axis=1)


scaler = StandardScaler()
scaler.fit(dc)
scaled_ds = pd.DataFrame(scaler.transform(dc),columns= dc.columns )
print("All features are now scaled")

print(scaled_ds.columns)

if not np.all(np.isfinite(scaled_ds)):
    #Option 2: Drop rows or columns with non-finite values
     scaled_ds = scaled_ds.dropna()  # Drop rows with NaN
    # scaled_ds.dropna(axis=1, inplace=True)  # Drop columns with NaN

# Elbow method implementataion:
from sklearn.cluster import KMeans

# Within-Cluster Sum of Squares (WCSS)

WCSS = [] 

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 43)
    kmeans.fit(scaled_ds) #applies k-means clustering to the dataset scaled_ds
    WCSS.append(kmeans.inertia_) #calculates and stores the within-cluster sum of squares (WCSS) for each number of clusters. The inertia_ attribute of the KMeans object gives the sum of squared distances of samples to their closest cluster center.
for i, val in enumerate(WCSS, 1):
    print(f'{i} : {val}')
    
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 43)
y_kmeans = kmeans.fit_predict(scaled_ds)

# Visualising the clusters
plt.scatter(scaled_ds.loc[y_kmeans == 0, 'Income'], scaled_ds.loc[y_kmeans == 0, 'Total_Spending'], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(scaled_ds.loc[y_kmeans == 1, 'Income'], scaled_ds.loc[y_kmeans == 1, 'Total_Spending'], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(scaled_ds.loc[y_kmeans == 2, 'Income'], scaled_ds.loc[y_kmeans == 2, 'Total_Spending'], s = 100, c = 'blue', label = 'Cluster 3')
plt.scatter(scaled_ds.loc[y_kmeans == 3, 'Income'], scaled_ds.loc[y_kmeans == 3, 'Total_Spending'], s = 100, c = 'yellow', label = 'Cluster 4')
plt.scatter(scaled_ds.loc[y_kmeans == 4, 'Income'], scaled_ds.loc[y_kmeans == 4, 'Total_Spending'], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'cyan', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Income')
plt.ylabel('Total_Spending')
plt.legend()
plt.show()


'''
Cluster 1 (Red): This cluster appears to be at the lower end of both income and spending. These customers may be categorized as low-income, low-spending.

Cluster 2 (Green): Customers in this cluster have a higher income compared to Cluster 1 but still have low spending. They could be considered as high-income, low-spending customers who are potentially conservative spenders or are not attracted to the current product offerings.

Cluster 3 (Blue): This group shows customers with mid-range income and spending habits. They represent the middle-market customers.

Cluster 4 (Yellow): These customers are also in the mid-range income bracket but have higher spending, suggesting that they might be less price-sensitive or more engaged with the offerings.

Cluster 5 (Magenta): This cluster is characterized by low income but high spending, indicating these customers may prioritize spending on the products or services being analyzed over other expenses, or they might be influenced by promotions.

Centroids (Cyan): Each cluster has a centroid (indicated by cyan dots), which is the mean position of all the points in the cluster. It acts as the "center" of each cluster.

The x-axis is labeled 'Income' and the y-axis 'Total_Spending'. Both axes have positive and negative values, suggesting that the data has been scaled or normalized.

From the perspective of a business, these clusters can help in tailoring marketing strategies. For instance, targeting might involve focusing on high-income, high-spending customers differently than low-income, high-spending customers. The company could also investigate why certain high-income customers are low-spenders and develop strategies to increase their spending.





'''

