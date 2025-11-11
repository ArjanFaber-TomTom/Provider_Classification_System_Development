import sys
import os
import json
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
from sklearn.metrics import silhouette_score
from datetime import date

threshold = 0.65 
dist_d = 0.2
dist_m = 0.8

def data_retrieval(threshold, df_providers_class_content, df_providers_class_service):
   
    df_providers_class_content = df_providers_class_content[df_providers_class_content.columns[0:-1]].dropna()
    df_providers_class_service = df_providers_class_service[df_providers_class_service.columns[0:-1]].dropna()
    print(df_providers_class_content.describe()), print(df_providers_class_service.describe())
    df_new = pd.DataFrame()

    row = []

    for i in range(len(df_providers_class_service)):
        for j in range(len(df_providers_class_content)):
            if df_providers_class_content["instance"][j][0:-1] in df_providers_class_service.loc[i, "server"] and df_providers_class_content["feed"][j] == df_providers_class_service["feed"][i] and df_providers_class_content["date"][j] == df_providers_class_service["date"][i]:
        
                # build row
                row = {
                 "date": df_providers_class_service.loc[i, "date"],
                "feed": df_providers_class_service.loc[i, "feed"],
                "server": df_providers_class_service.loc[i, "server"],
                "map_matching": df_providers_class_content.loc[j, "map_matching"],
                "exclusive": df_providers_class_content.loc[j, "exclusive"],
                "redundant": df_providers_class_content.loc[j, "redundant"],
                "accidents": df_providers_class_service.loc[i, "accidents"],
                "hazards": df_providers_class_service.loc[i, "hazards"],
                "closures": df_providers_class_service.loc[i, "closures"],
                "freshness": df_providers_class_service.loc[i, "freshness"],
                "contribution": df_providers_class_service.loc[i, "contribution"],
                "usability": df_providers_class_service.loc[i, "usability"],
                "availability": df_providers_class_service.loc[i, "availability"],

                }
                row = pd.DataFrame([row])
                # append row to dataframe
                df_new = pd.concat([df_new, row])
          
        print("cycle i:", i, " out of ", len(df_providers_class_service))

    merged_df = df_new
    df_complete = merged_df
    FEATURES = ['map_matching', 'contribution', 'redundant', 'freshness', 'usability', 'availability']


    for col in FEATURES:
        df_complete[col] = pd.to_numeric(df_complete[col], errors='coerce')
    df_complete['freshness'] = (df_complete['freshness'] - df_complete['freshness'].min()) / (df_complete['freshness'].max() - df_complete['freshness'].min())
   
    #corr_matrix = df_sub.corr()
    #plt.figure(figsize=(10, 8))
    #sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    #plt.title('Correlation Heatmap')
    #plt.savefig("corr_matrix.png")
    #plt.show()

    # Extract attributes with absolute correlation above 0.65 (excluding self-correlation)
    #high_corr_cols = set()
    
    #for col in corr_matrix.columns:
    #    for idx in corr_matrix.index:
    #        if col != idx and abs(corr_matrix.loc[idx, col]) > threshold:
    #            high_corr_cols.add(col)
    #            high_corr_cols.add(idx)

    # Convert to list
    #high_corr_cols = list(high_corr_cols)
    #print("Attributes with absolute correlation above 0.65:", high_corr_cols)

    # Optional: store these attributes in a new dataframe
    #df_high_corr = df_sub[high_corr_cols]

    #df_subset = df_sub

    # Plot the average value for each column
    #df_subset.mean().plot(kind='bar', figsize=(12, 6))
    #plt.ylabel('Average value')
    #plt.title('Average Scores per Attribute')
    #plt.xticks(rotation=45)
    #plt.grid(True)
    #plt.tight_layout()
    #plt.savefig("avg_attributes.png")
    #plt.show()

    return df_complete

from sklearn.mixture import GaussianMixture

def cluster_gmm(df, a=1,b=1.960, grades = ['A+', 'A', 'B', 'C', 'D']):
    FEATURES = ['map_matching', 'contribution', 'redundant', 'freshness', 'usability', 'availability']

    df = df[FEATURES]

    # --- Fit GMM ---
    gmm = GaussianMixture(n_components=len(grades), covariance_type='full', random_state=42)
    df['cluster_GMM'] = gmm.fit_predict(df)

    # --- Probabilities for each cluster ---
    proba = gmm.predict_proba(df.drop(columns=['cluster_GMM']))
    df = df.copy()  # before any assignment
    df.loc[:, [f'cluster_{i}_prob' for i in range(5)]] = proba

    # --- Cluster summary
    cluster_summary = df.groupby('cluster_GMM')[FEATURES].agg(['mean', 'min', 'max', 'count'])

    mean_cols = cluster_summary.loc[:, pd.IndexSlice[:, 'mean']]
    mean_cols.columns = mean_cols.columns.droplevel(1)

    # Average mean score
    cluster_summary['average_mean'] = mean_cols.mean(axis=1)

    # Variance score
    cluster_variance = df.groupby('cluster_GMM')[FEATURES].var()
    cluster_summary['variance_score'] = cluster_variance.mean(axis=1)

    # --- Extract means for custom scoring ---
    map_matching_mean = cluster_summary[('map_matching', 'mean')]
    contribution_mean = cluster_summary[('contribution', 'mean')]
    redundant_mean = cluster_summary[('redundant', 'mean')]
    availability_mean = cluster_summary[('availability', 'mean')]
    freshness_mean = cluster_summary[('freshness', 'mean')]
    usability_mean = cluster_summary[('usability', 'mean')]

    # --- Custom score formula ---
    cluster_summary['custom_score'] = (
        map_matching_mean  + availability_mean +
        freshness_mean + contribution_mean + usability_mean - redundant_mean  ) 
    # Rank clusters by custom score
    cluster_summary['custom_rank'] = cluster_summary['custom_score'].rank(ascending=False)
    cluster_summary_mean = cluster_summary.sort_values('custom_rank', ascending=True)

    # Assign grades (custom score)
    
    cluster_summary_mean['grade'] = pd.Series(grades[:len(cluster_summary_mean)],
                                          index=cluster_summary_mean.index)

    # --- Rank by variability ---
    cluster_summary_var = cluster_summary.sort_values('variance_score', ascending=True)
    cluster_summary_var['grade'] = pd.Series(grades[:len(cluster_summary_var)],
                                         index=cluster_summary_var.index)

    rows_lb = []
    rows_ub = []
    for i in range(len(grades)):
      rows_lb.append(a * cluster_summary_mean.custom_score[i] - b * np.sqrt(cluster_summary_var.variance_score[i]) )
    cluster_summary['threshold_lb'] = rows_lb

    for i in range(len(grades)):
      rows_ub.append(a * cluster_summary_mean.custom_score[i] + b * np.sqrt(cluster_summary_var.variance_score[i]) )
    cluster_summary['threshold_ub'] = rows_ub

    # Rank by threshold
    cluster_summary_thld = cluster_summary.sort_values('threshold_ub', ascending=False)
    cluster_summary_thld['grade'] = pd.Series(grades[:len(cluster_summary_thld)],
                                          index=cluster_summary_thld.index)

    score_pca = silhouette_score(df, df['cluster_GMM'])
    bic = gmm.bic(df[FEATURES])
    aic = gmm.aic(df[FEATURES])
    
    return df, cluster_summary_mean[['custom_score', 'grade']], cluster_summary_var[['variance_score', 'grade']], cluster_summary_thld[['threshold_lb','threshold_ub', 'grade']].sort_index(), score_pca, bic,aic


def market_requirements(grades = ["D", "C", "B", "A", "A+"]):

    market_req = pd.DataFrame({"Accuracy_lb": [0, 0.15, 0.5,0.85,0.95],"Completeness_lb": [0,0.10,0.4,0.6,0.90],"Correctness_lb": [0,0.15,0.5,0.85,0.95], "Availability_lb": [0,0.85,0.95,0.97,0.99], "Timelineness_lb": [0,0.15,0.8,0.85,0.95], "Usability_lb": [0,0.15,0.5,0.85,0.95] , 
                              "Accuracy_ub": [0.15, 0.50, 0.85,0.95,1],"Completeness_ub": [0.1,0.4,0.6,0.9,1],"Correctness_ub": [0.15,0.5,0.85,0.95,1], "Availability_ub": [0.85,0.95,0.97,0.99,1], "Timelineness_ub": [0.15,0.8,0.85,0.95,1], "Usability_ub": [0.15,0.5,0.85,0.95,1] })
    market_req['Classification'] = ["D", "C", "B", "A", "A+"]

    
    columns = ['Accuracy_lb', 'Completeness_lb', 'Correctness_lb', 'Availability_lb',
       'Timelineness_lb', 'Usability_lb']
    market_req_thld_lb = []
    market_req_thld_ub = []

    for idx in range(len(grades)):
        row_lb = 0
        row_ub = 0

        for j in range(len(columns)):
            row_lb += market_req[market_req.columns[j]][market_req['Classification']==grades[idx]] 
            row_ub += market_req[market_req.columns[j + len(grades)]][market_req['Classification']==grades[idx]]

        market_req_thld_lb.append(row_lb)
        market_req_thld_ub.append(row_ub)

    
    return market_req_thld_lb, market_req_thld_ub

def add_attribute_grades(df, grades=["D", "C", "B", "A", "A+"]):
    # Define thresholds for each attribute
    market_req = pd.DataFrame({
        "Accuracy_lb": [0, 0.15, 0.5, 0.85, 0.95],
        "Completeness_lb": [0, 0.10, 0.4, 0.6, 0.90],
        "Correctness_lb": [0, 0.15, 0.5, 0.85, 0.95],
        "Availability_lb": [0, 0.85, 0.95, 0.97, 0.99],
        "Timelineness_lb": [0, 0.15, 0.8, 0.85, 0.95],
        "Usability_lb": [0, 0.15, 0.5, 0.85, 0.95],
        "Accuracy_ub": [0.15, 0.50, 0.85, 0.95, 1],
        "Completeness_ub": [0.1, 0.4, 0.6, 0.9, 1],
        "Correctness_ub": [0.15, 0.5, 0.85, 0.95, 1],
        "Availability_ub": [0.85, 0.95, 0.97, 0.99, 1],
        "Timelineness_ub": [0.15, 0.8, 0.85, 0.95, 1],
        "Usability_ub": [0.15, 0.5, 0.85, 0.95, 1]
    })
    market_req['Classification'] = grades

    # Map DataFrame attribute to market requirements columns
    attribute_map = {
        "map_matching": ("Accuracy_lb", "Accuracy_ub"),
        "contribution": ("Correctness_lb", "Correctness_ub"),
        "redundant": ("Timelineness_lb", "Timelineness_ub"),  # lower is better
        "freshness": ("Completeness_lb", "Completeness_ub"),
        "usability": ("Usability_lb", "Usability_ub"),
        "availability": ("Availability_lb", "Availability_ub")
    }

    def get_grade(attr, value):
        lb_col, ub_col = attribute_map[attr]
        # For redundant, lower value is better
        if attr == "redundant":
            value = 1 - value
        for i, grade in enumerate(grades):
            lb = market_req[lb_col][i]
            ub = market_req[ub_col][i]
            if lb <= value <= ub:
                return grade
        return "D"

    # Add a new column for each attribute grade
    for attr in attribute_map.keys():
        df[f"{attr}_grade"] = df[attr].apply(lambda x: get_grade(attr, x))

    return df



def thresholds_fun(model, df, w_data = 0.25, w_market= 0.75, grades = ['D', 'C', 'B', 'A', 'A+']):
    
    df_clus, df_clus_mean, df_clus_var, df_clus_thld, score_gmm, bic, aic =  cluster_gmm(df)
  
    var_name = 'cluster_GMM'
    thld_market_req_lb,thld_market_req_ub = market_requirements(grades)
    df_clus_thld = df_clus_thld.sort_values(['threshold_ub'], ascending=True)
    flat_list_ub = [x.iloc[0] for x in thld_market_req_ub] 
    flat_list_ub = [float(x) for x in flat_list_ub]
    df_clus_thld['threshold_ub'] = w_data * df_clus_thld['threshold_ub']  + w_market * np.array(flat_list_ub)
    flat_list_lb = [x.iloc[0] for x in thld_market_req_lb] 
    flat_list_lb = [float(x) for x in flat_list_lb]
    df_clus_thld['threshold_lb'] = w_data * df_clus_thld['threshold_lb']  + w_market * np.array(flat_list_lb)
    

    return df_clus_thld

def analyze_cluster(cluster_id, df_mean, df_var, df_thld):
    # Make sure we're working with single rows as Series
    mean_row = df_mean.sort_index().loc[cluster_id]
    var_row = df_var.sort_index().loc[cluster_id]
    thld_row = df_thld.sort_index().loc[cluster_id]
    
    recommendations = []
     
    # Custom score
    val_mean = mean_row['grade'].values
    if val_mean in ['D', 'C', 'B']:
        recommendations.append(
        f"Increase custom_score from cluster average {mean_row['custom_score'].values[0]:.3f} to improve grade ({mean_row['grade'].values[0]} → higher)."
        )

    val_var = var_row['grade'].values
    # Variance score
    if val_var in ['D', 'C', 'B']:
        recommendations.append(
            f"Reduce variance_score from current cluster average variance of {var_row['variance_score'].values[0]:.5f} to improve stability ({var_row['grade'].values[0]} → higher)."
            )

    # Threshold
    val_thld = thld_row['grade'].values
    if val_thld in ['D', 'C', 'B']:
        recommendations.append(
            f"Increase thresholds from (lower bound - upper bound) {thld_row['threshold_lb'].values[0]:.3f}-{thld_row['threshold_ub'].values[0]:.3f} to improve your threshold grade ({thld_row['grade'].values[0]} → higher)."
        )


    if not recommendations:
            recommendations.append("Cluster is performing very well across all metrics (A/A+).")
    
    return recommendations

def cluster_improvement_plan(df_mean, df_var, df_thld):
    plan = {}
    for clus in df_mean.index:
        plan[clus] = analyze_cluster(clus, df_mean, df_var, df_thld)
    return plan


def create_data(cluster, df, df_clus_mean,df_clus_var,df_clus_thld, dist_d,dist_m, feed,server,grade):
    
   #Weight fixed market requirements:** 
    grades = ['A+', 'A', 'B', 'C', 'D']
    feed = feed
    server = server
    feed_date = df['date']
    today_date = date.today()
    grade = grade
    grade_group = cluster

    md = ""
    plan = cluster_improvement_plan(df_clus_mean, df_clus_var, df_clus_thld)
    for clus, recs in plan.items():
        if clus == cluster:
            md += f"\n\n Group {clus} ( {grade}) Recommendations:\n"
            for r in recs:
                md += f"- {r}\n"

    rec = md 
    map_matching = df['map_matching']
    contribution = df['contribution']
    redundant = df['redundant']
    freshness = df['freshness']
    usability = df['usability']
    availability = df['availability']

    sample_df = pd.DataFrame({
    "map_matching": [map_matching],
    "contribution": [contribution],
    "redundant": [redundant],
    "freshness": [freshness],
    "usability": [usability],
    "availability": [availability]
})
    sample_df
    df_with_grades = add_attribute_grades(sample_df)
    print (df_with_grades)
    try:

        conn = psycopg2.connect(
            dbname="tti_database",
            user="YE2137_user_tti_quality",
            password="'9GV~NGU3uUZL8v",
            host="tti-biba-db.postgres.database.azure.com",
            port=5432,
            sslmode="require"
        )
        cur = conn.cursor()
    
        insert_query = """
    INSERT INTO ba.bahamas 
    (today_date, feed_date, feed, server, grade, dist_d, dist_m, grade_group, recommendation, 
    map_matching, contribution, redundant, freshness, usability, availability,
    map_matching_grade, contribution_grade, redundant_grade, freshness_grade, usability_grade, availability_grade)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s,%s,%s,%s,%s)
    """
        map_matching_grade = df_with_grades['map_matching_grade'].values[0]
        contribution_grade = df_with_grades['contribution_grade'].values[0]
        redundant_grade = df_with_grades['redundant_grade'].values[0]
        freshness_grade = df_with_grades['freshness_grade'].values[0]
        usability_grade = df_with_grades['usability_grade'].values[0]
        availability_grade = df_with_grades['availability_grade'].values[0]
        
        data_tuple = (
        today_date,
        feed_date,
        str(feed),
        str(server),
        str(grade),
        float(dist_d),
        float(dist_m),
        int(grade_group),
        str(rec),
        float(map_matching),
        float(contribution),
        float(redundant),
        float(freshness),
        float(usability),
        float(availability),
        str(map_matching_grade),
        str(contribution_grade),
        str(redundant_grade),
        str(freshness_grade),
        str(usability_grade),
        str(availability_grade)
        )

        cur.execute(insert_query, data_tuple)
        conn.commit()

        print("✅ Report data (with recommendations) inserted successfully into ba.bahamas")

        cur.close()
        conn.close()

    except Exception as e:
        print("❌ Database insert failed:", e)

def assign_cluster(score):
    for cluster, row in thresholds.iterrows():
        if row['threshold_lb'] <= score <= row['threshold_ub']:
            return cluster
    return None  # if not within any threshold range
    
pw = "'9GV~NGU3uUZL8v"


try:
    conn = psycopg2.connect(
        dbname="tti_database",
        user="YE2137_user_tti_quality",
        password=pw, 
        host="tti-biba-db.postgres.database.azure.com",
        port=5432,
        sslmode="require"
    )
    print("Connection successful!")

    cur = conn.cursor()


    # --- Step 1: Import data from the two tables ---
    query_content = "SELECT * FROM ba.content_overview_new;"
    query_service = "SELECT * FROM ba.service_overview_new;"

    df_content = pd.read_sql_query(query_content, conn)
    df_service = pd.read_sql_query(query_service, conn)
   
    df = data_retrieval(0.65, df_content, df_service)
    FEATURES = ['map_matching', 'contribution', 'redundant', 'freshness', 'usability', 'availability']

    df_clus, df_clus_mean, df_clus_var, df_clus_thld, score_gmm, bic, aic =  cluster_gmm(df[FEATURES])
    thresholds = pd.DataFrame({
    'threshold_lb': df_clus_thld.threshold_lb,
    'threshold_ub': df_clus_thld.threshold_ub,
    })
    df['custom_score_raw'] =  df['map_matching'] +  df['exclusive']  + df['availability'] + df['freshness'] + df['contribution'] + df['usability'] - df['redundant']
    df['cluster_GMM'] = df['custom_score_raw'].apply(assign_cluster)
    feeds = df['feed']
    servers = df['server']
    df[['custom_score_raw', 'cluster_GMM']].reset_index()

    cur.execute("""
    DROP TABLE IF EXISTS ba.bahamas;
    CREATE TABLE IF NOT EXISTS  ba.bahamas (   
    today_date DATE,
    feed_date DATE,
    feed TEXT,
    server TEXT,
    grade TEXT,
    dist_d FLOAT,
    dist_m FLOAT,
    grade_group INT,           
    recommendation TEXT,
    map_matching FLOAT,
    contribution FLOAT,
    redundant FLOAT,
    freshness FLOAT,
    usability FLOAT,
    availability FLOAT,
    map_matching_grade TEXT,
    contribution_grade TEXT,
    redundant_grade TEXT,
    freshness_grade TEXT,
    usability_grade TEXT,
    availability_grade TEXT        
                            
    );

        """)
    conn.commit()


    cur = conn.cursor()
    for i in range(len(df)):

        cluster = float(df['cluster_GMM'].values[i])
        if np.isnan(df['cluster_GMM'].values[i]):
            continue
        model = 0
        df_clus_thld = thresholds_fun(model,df[df.columns[2:]].reset_index(drop=True),dist_d, dist_m)
        grade = df_clus_thld.sort_index()['grade'][cluster]
        feed = df['feed'].values[i]
        server = df['server'].values[i]
        create_data(cluster, df.iloc[i], df_clus_mean, df_clus_var, df_clus_thld.sort_index(),dist_d, dist_m,feed,server,grade)

    cur.close()
    conn.close()
except Exception as e:
    print("Connection failed:", e)
