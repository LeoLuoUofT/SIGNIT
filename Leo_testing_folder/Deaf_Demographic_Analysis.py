from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

spark = SparkSession.builder.appName("Hearing Loss Analysis").getOrCreate()

path = "/Users/akankshasathish/Downloads/13100748new.csv"
df = spark.read.csv(path, header=True, inferSchema=True).cache()

df = df.withColumn("VALUE", col("VALUE").cast("float"))

df.createOrReplaceTempView("hearing_loss_data")

age_sex_query = spark.sql("""
    SELECT Age_group, Sex, AVG(VALUE) as avg_hearing_loss
    FROM hearing_loss_data
    WHERE Categories = 'Hearing loss'
    Group BY Age_group, Sex
    ORDER BY Age_group, Sex
""")
print("Age group and Sex Query results:")
age_sex_query.show()
age_sex_data = age_sex_query.toPandas()

plt.figure(figsize=(12, 6))
sns.barplot(x= 'Age_group', y='avg_hearing_loss', hue='Sex', data=age_sex_data)
plt.title('Average Hearing Loss by Age Group and Sex')
plt.xlabel('Age Group')
plt.ylabel('Average Hearing Loss')
plt.xticks(rotation=45)
plt.show()

rate_of_change_query = spark.sql("""
    SELECT Age_group,
      (MAX(VALUE) - MIN(VALUE)) / COUNT(*) as rate_of_change
    FROM hearing_loss_data
    WHERE Categories = 'Hearing loss'
    GROUP BY Age_group
    ORDER BY Age_group
""")
print("Rate of Change Query results:")
rate_of_change_query.show()
rate_of_change_data = rate_of_change_query.toPandas()

plt.figure(figsize=(12,6))
plt.stackplot(rate_of_change_data['Age_group'], rate_of_change_data['rate_of_change'], labels=['Rate of Change'], colors=['#ff9999'])
plt.title('Rate of Change in Hearing Loss by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Rate of Change')
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.show()

year_query = spark.sql("""
    SELECT REF_DATE, AVG(VALUE) as avg_hearing_loss
    FROM hearing_loss_data
    WHERE Categories = 'Hearing loss'
    GROUP BY REF_DATE
    ORDER BY REF_DATE
""")
print("Year Query Results:")
year_query.show()
year_data = year_query.toPandas()

plt.figure(figsize=(12, 6))
sns.lineplot(x='REF_DATE', y='avg_hearing_loss', data=year_data)
plt.title('Average Hearing Loss by year')
plt.xlabel('year')
plt.ylabel('Average Hearing Loss')
plt.xticks(rotation=45)
plt.show()

geo_query = spark.sql("""
    SELECT GEO, AVG(VALUE) as avg_hearing_loss
    FROM hearing_loss_data
    WHERE Categories = 'Hearing loss'
    GROUP BY GEO
    ORDER BY avg_hearing_loss DESC
""")
print("Geography Query Results:")
geo_query.show()
geo_data = geo_query.toPandas()

plt.figure(figsize=(12,6))
sns.barplot(x='avg_hearing_loss', y='GEO', data=geo_data)
plt.title('Average hearing Loss by Geography')
plt.xlabel('Average Hearing Loss')
plt.ylabel('Geography')
plt.show()

confidence_query = spark.sql("""
    SELECT REF_DATE, AVG(VALUE) as avg_estimate,
           AVG(CASE WHEN Characteristics LIKE '%Low%' THEN VALUE END) as avg_low_confidence,
           AVG(CASE WHEN Characteristics LIKE '%High%' THEN VALUE END) as avg_high_confidence
    FROM hearing_loss_data
    WHERE Categories = 'Hearing loss'
    GROUP BY REF_DATE
    ORDER BY REF_DATE 
 """)
print("Confidence Query Results:")
confidence_query.show()
confidence_data = confidence_query.toPandas()

plt.figure(figsize=(12, 6))
sns.lineplot(x='REF_DATE', y='avg_estimate', data=confidence_data, label='Average Estimate')
sns.lineplot(x='REF_DATE', y='avg_low_confidence', data=confidence_data, label='Low Confidence')
sns.lineplot(x='REF_DATE', y='avg_high_confidence', data=confidence_data, label='High Confidence')
plt.title('Average Estimate with Confidence Levels by Year')
plt.xlabel('Year')
plt.ylabel('Average Estimate')
plt.xticks(rotation=45)
plt.legend()
plt.show()

data_query = spark.sql("""
      SELECT  Age_group, Sex, VALUE
      FROM hearing_loss_data
      WHERE Categories = 'Hearing loss'
""")
info_df = data_query.toPandas()

info_df['VALUE'].replace('',np.nan, inplace=True)
info_df.dropna(subset=['VALUE'], inplace=True)

def calculate_variance(values):
    mean_data = sum(values) / len(values)
    return sum((x - mean_data) ** 2 for x in values) / len(values)

def calculate_std_dev(variance):
    return math.sqrt(variance)

results = []
 
grouped_data = info_df.groupby(['Age_group', 'Sex'])

for (age_group, sex), group in grouped_data :
    values = group['VALUE'].tolist()
    if len(values) > 1:
        variance = calculate_variance(values)
        std_dev = calculate_std_dev(variance)
        mean_data = sum(values) / len(values)
        if not math.isnan(variance) and variance > 0:
            cv = std_dev / mean_data
            results.append({
                'Age_group' : age_group,
                'Sex': sex,
                'Variance': variance,
                'Std_Deviation' : std_dev,
                'Mean' : mean_data,
                'CV' : cv
            })
results_df = pd.DataFrame(results)
print(results_df)

plt.figure(figsize=(14, 8))
ax = sns.barplot(x='Age_group', y='Variance', hue='Sex', data=results_df, palette='pastel')
ax2 = ax.twinx()
sns.lineplot(x='Age_group', y='Std_Deviation', hue='Sex', data=results_df, marker='o', ax=ax2)
ax.set_title('Varinace and Standard Deviation of Hearing Loss by Age Group and Sex')
ax.set_xlabel('Age Group')
ax.set_ylabel('Variance of Hearing Loss')
ax2.set_ylabel('Standard Deviation of Hearing Loss')
ax.legend(title='Sex', loc='upper left')
ax2.legend(title= 'Sex', loc='upper right')
plt.show()

pivot_table = results_df.pivot(index="Age_group", columns="Sex", values="CV")
plt.figure(figsize=(12,8))
sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('heatmap of Coefficient of Variation of Hearing Loss by Age Group and Sex')
plt.ylabel('Age Group')
plt.xlabel('Sex')
plt.show()

Quartile_hearing_query = spark.sql("""
            SELECT Age_group,
                percentile_approx(VALUE, 0.25) as Q1_hearing_loss,
                percentile_approx(VALUE, 0.50) as median_hearing_loss,
                percentile_approx(VALUE, 0.75) as Q3_hearing_loss
FROM hearing_loss_data
WHERE Categories = 'Hearing loss'
GROUP BY Age_group
ORDER BY Age_group
""")
print("Quartile Hearing Query Results: ")
Quartile_hearing_query.show()
Quartile_hearing_data = Quartile_hearing_query.toPandas()

plt.figure(figsize=(12,8))
sns.lineplot(data=Quartile_hearing_data, x='Age_group', y='Q1_hearing_loss', marker='o', label='Q1')
sns.lineplot(data=Quartile_hearing_data, x='Age_group', y='median_hearing_loss', marker='o', label='Median')
sns.lineplot(data=Quartile_hearing_data, x='Age_group', y='Q3_hearing_loss', marker='o', label='Q3')
plt.title('Quartile hearing Loss by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Hearing Loss')
plt.legend()
plt.show()







                                                                      
