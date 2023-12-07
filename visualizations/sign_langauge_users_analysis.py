from pyspark.sql import SparkSession
import plotly.express as px
import pandas as pd

spark = SparkSession.builder.appName("Deaf Sign Language Analysis").getOrCreate()

deaf_signlanguage_df = spark.read.format("csv").option("header", "true").load("/Users/akankshasathish/Downloads/Deaf_signlanguage.csv")

deaf_signlanguage_df.createOrReplaceTempView("deaf_signlanguage")

deaf_users_query = spark.sql("""
 SELECT
    SUM(CAST(Yes AS INT)) AS deaf_signlanguage_users,
    ROUND((SUM(CAST(Yes AS INT)) / COUNT(*)) * 100, 2) AS percentage
 FROM deaf_signlanguage
""")

deaf_users_result = deaf_users_query.collect()[0]
print(f"Deaf Sign Language users: {deaf_users_result['deaf_signlanguage_users']}, percentage: {deaf_users_result['percentage']}%")

df_1st_query = deaf_signlanguage_df.toPandas()

fig_1st_query = px.bar(df_1st_query, x="Deaf and uses sign language - total responses", y="% Yes", title="Deaf Sign Language Users percentage")
fig_1st_query.update_xaxes(title_text=" Deaf Demographics")
fig_1st_query.update_yaxes(title_text=" Sign language Usage Rate")
fig_1st_query.show()
top_regions_query = spark.sql("""
    SELECT
        `SHA name` AS region_name,
         SUM(`Deaf and uses sign language - total responses`) AS total_respondents,
         SUM(CAST(Yes AS INT)) AS deaf_signlanguage_users
    FROM deaf_signlanguage
    GROUP BY region_name
    ORDER BY deaf_signlanguage_users DESC
    LIMIT 5
""")

top_regions = top_regions_query.collect()

regions = [row['region_name'] for row in top_regions]
total_respondents = [row['total_respondents'] for row in top_regions]
deaf_users = [row['deaf_signlanguage_users'] for row in top_regions]

for row in top_regions:
    print(f"REGION: {row['region_name']}, Total: {row['total_respondents']}, Deaf Users: {row['deaf_signlanguage_users']}")

df_2nd_query = top_regions_query.toPandas()
sorted_df = df_2nd_query.sort_values(by = 'deaf_signlanguage_users', ascending=False).head(5)
fig_2nd_query = px.scatter(sorted_df, x='total_respondents', y='deaf_signlanguage_users',
                           size='deaf_signlanguage_users', color='region_name',
                           hover_name='region_name',
                           labels={'deaf_signlanguage_users':'Deaf Sign Language Users', 'total_respondents':'Total Respondents'},
                            title='Top 5 Regions by Deaf Sign Language Users')
fig_2nd_query.update_layout(
    xaxis_title='Total Respondents',
    yaxis_title='Number of Deaf Sign Language Users',
    showlegend=True
)

fig_2nd_query.show()
regional_data_query = spark.sql("""
    SELECT
        `SHA name`,
         SUM(CAST(Yes AS INT)) AS yes_count,
         SUM(CAST(No AS INT)) AS no_count
    FROM deaf_signlanguage
    GROUP BY `SHA name`
    HAVING SUM(CAST(Yes AS INT)) > 0 OR SUM(CAST(No AS INT)) > 0
""")

regional_data = regional_data_query.collect()

for row in regional_data:
    print(f"Region Name: {row['SHA name']}, Deaf Individuals who use sign Langauge: {row['yes_count']}, Non-Deaf Individuals who do not use sign Langauge: {row['no_count']}")

df_3rd_query = regional_data_query.toPandas()

fig_3rd_query = px.area(df_3rd_query, x='yes_count', y='no_count', color='SHA name',
                title='Deaf Sign Language Users by Region',
                labels={'yes_count': 'Deaf Individuals who use sign Language', 'no_count': 'Non-Deaf Individuals who do not use sign Language'})
fig_3rd_query.show()

spark.stop()
