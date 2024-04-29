import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("SocialNetworkAds.csv")
print(df.head(10))
print(df.dtypes)
print(df.describe())
print(df.shape)
print(df.Gender.astype('category').cat.codes)

def DetectOutlier(df, var):
	q1 = df[var].quantile(0.25)
	q3 = df[var].quantile(0.75)
	iqr = q3 - q1
	high, low = q3 + 1.5 * iqr, q1 - 1.5 * iqr
	print("\nhighest allowed in variables", var, high)
	print("\nlowest allowed in variables", var, low)
	count = df[(df[var] > high) | (df[var] < low)][var].count()
	print("\ntotal outlier in {}: {}".format(var, count))
	df_no_outliers = df[((df[var] >= low) & (df[var] <= high))]
	print("\noutliers removed in", var)
	return df_no_outliers

df_numeric = df.drop(columns=['Gender'])  

df = DetectOutlier(df_numeric, 'Age')

sns.heatmap(df.corr(), annot=True)
plt.show()

