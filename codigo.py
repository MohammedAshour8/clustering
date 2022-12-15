#!/usr/bin/env python3



import os.path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sqlalchemy
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn_som.som import SOM



def remove_outliers(df):
    """Function to remove outliers"""
    
     # Calculate the Q3 and Q1
    Q3 = df.quantile(0.75, numeric_only=True)
    Q1 = df.quantile(0.25, numeric_only=True)
    
    # Calculate the IQR
    IQR = Q3 - Q1
    
    # Calculate the upper and lower limits of the dataframe
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    
    # Add index of rows that are outliers to a list
    indexesToDelete = []
    for i in range(len(df)):
        for j in range(len(df.columns)):
            if df.iloc[i, j] > upper_limit[j] or df.iloc[i, j] < lower_limit[j]:
                indexesToDelete.append(i)

    # Drop from the dataframe the rows whose indexes are in the list
    df.drop(indexesToDelete, inplace=True)

    return df



def main():
    """Main function"""

    # VARIABLES
    input_csv = 'AMD_clustering_0.csv'
    processed_csv = 'AMD_clustering_0_no_outliers.csv'
    csv_percentage_data = 1.0



    # DATASET FROM CSV FILE
    # Check if the input CSV exists; if not, exit
    if not os.path.exists(input_csv):
        print("The input file \"" + input_csv + "\" doesn't exist.")
        exit()


    # Check if the processed CSV exists; if yes, read it; if not, read the input CSV and process it
    if not os.path.isfile(processed_csv):
        # Read the data with outliers
        csv_df = pd.read_csv(input_csv, index_col=False)

        # Remove the outliers and save the processed dataframe to a CSV file
        csv_df = remove_outliers(csv_df)
        csv_df.to_csv(processed_csv, index=False)
    else:
        # Read the data without outliers
        csv_df = pd.read_csv(processed_csv, index_col=False)


    # Get the indicated percentage of the data
    csv_data_sample = csv_df.iloc[0:int(len(csv_df) * csv_percentage_data / 100)]


    # K-means
    # Elbow method
    wcss = []
    for i in range(1, 11):
        modelo = KMeans(n_clusters=i)
        agrup = modelo.fit(csv_data_sample)
        wcss.append(agrup.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Método do cóbado para K-means')
    plt.xlabel('Número de clusters')
    plt.ylabel('Custo')
    plt.savefig('csv_kmeans_elbow.png')
    plt.clf()

    # Model
    model = KMeans(n_clusters=3)
    agrup1 = model.fit(csv_data_sample)
    csv_kmeans_cp = csv_data_sample.copy()
    csv_kmeans_cp['cluster'] = agrup1.labels_.tolist()
    sns.pairplot(csv_kmeans_cp, hue='cluster', palette = "muted")
    plt.savefig('csv_kmeans.png')
    plt.clf()
    

    # SOM
    data = csv_data_sample.values

    # Elbow method
    wcss = []
    for i in range(1, 11):
        data_som = SOM(m=i, n=1, dim=len(csv_data_sample.columns))
        data_som.fit(data)
        wcss.append(data_som.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Método do cóbado para SOM')
    plt.xlabel('Número de clusters')
    plt.ylabel('Custo')
    plt.savefig('csv_som_elbow.png')
    plt.clf()

    # Model
    data_som = SOM(m=3, n=1, dim=len(csv_data_sample.columns))
    data_som.fit(data)
    csv_som_cp = csv_data_sample.copy()
    csv_som_cp['cluster'] = data_som.predict(data)
    sns.pairplot(csv_som_cp, hue='cluster', palette="muted")
    plt.savefig('csv_som.png')
    plt.clf()


    # DBSCAN
    estimated_minPts = 2 * csv_data_sample.shape[1]

    # Elbow method
    neighbors = NearestNeighbors(n_neighbors=estimated_minPts)
    neighbors_fit = neighbors.fit(csv_data_sample)
    distances, _ = neighbors_fit.kneighbors(csv_data_sample)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.xticks(np.arange(28000, 30000, 500))
    plt.xlim(28000, 30000)
    plt.yticks(np.arange(0.0, 0.01, 0.001))
    plt.ylim(0.0, 0.01)
    plt.plot(distances)
    plt.savefig('csv_dbscan_elbow.png')
    plt.clf()

    estimated_eps = 0.005

    # Algorithm
    model = DBSCAN(eps=estimated_eps, min_samples=estimated_minPts)
    csv_dbscan_cp = csv_data_sample.copy()
    csv_dbscan_cp['cluster'] = model.fit_predict(csv_dbscan_cp.to_numpy())
    sns.pairplot(csv_dbscan_cp, hue='cluster', diag_kws={"hue": None, "color" : "0.2"})
    plt.savefig('csv_dbscan.png')
    plt.clf()



    # DATASET FROM POSTGRESQL DATABASE
    # Connect to the PostgreSQL database
    db_engine = sqlalchemy.create_engine("postgresql+psycopg2://postgres:postgres@172.17.0.2:5432/P7")
    connection = db_engine.connect()
    
    # Get the data from the database
    db_df = pd.read_sql_query('SELECT * FROM databank_world_deployment_indicators', connection)

    # Close the database connection
    connection.close()
    db_engine.dispose()

    # Clean the data
    db_df = db_df.drop(columns=['time', 'time_code', 'country_name', 'country_code'])
    db_df = db_df.interpolate()
    db_df = remove_outliers(db_df)


    # K-means
    # Elbow method
    wcss = []
    for i in range(1, 11):
        modelo = KMeans(n_clusters=i)
        agrup = modelo.fit(db_df)
        wcss.append(agrup.inertia_)
    plt.clf()
    plt.plot(range(1, 11), wcss)
    plt.title('Método do cóbado para K-means')
    plt.xlabel('Número de clusters')
    plt.ylabel('Custo')
    plt.savefig('db_kmeans_elbow.png')
    plt.clf()
    
    # Model
    model = KMeans(n_clusters=5)
    agrup2 = model.fit(db_df)
    db_kmeans_cp = db_df.copy()
    db_kmeans_cp['cluster'] = agrup2.labels_.tolist()
    sns.pairplot(db_kmeans_cp, hue='cluster', palette = "muted")    
    plt.savefig('db_kmeans.png')
    plt.clf()


    # SOM
    data = db_df.values

    # Elbow method
    wcss = []
    for i in range(1, 11):
        data_som = SOM(m=i, n=1, dim=db_df.shape[1])
        data_som.fit(data)
        wcss.append(data_som.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Método do cóbado para SOM')
    plt.xlabel('Número de clusters')
    plt.ylabel('Custo')
    plt.savefig('db_som_elbow.png')
    plt.clf()

    # Model
    data_som = SOM(m=5, n=1, dim=db_df.shape[1])
    data_som.fit(data)
    db_som_cp = db_df.copy()
    db_som_cp['cluster'] = data_som.predict(data)
    sns.pairplot(db_som_cp, hue='cluster', palette="muted")
    plt.savefig('db_som.png')
    plt.clf()


    # DBSCAN
    estimated_minPts = 2 * db_df.shape[1]

    # Elbow method
    neighbors = NearestNeighbors(n_neighbors=estimated_minPts)
    neighbors_fit = neighbors.fit(db_df)
    distances, _ = neighbors_fit.kneighbors(db_df)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.xticks(np.arange(225, 350, 25))
    plt.xlim(225, 350)
    plt.plot(distances)
    plt.savefig('db_dbscan_elbow.png')
    plt.clf()

    estimated_eps = 3000

    # Model
    model = DBSCAN(eps=estimated_eps, min_samples=estimated_minPts)
    db_dbscan_cp = db_df.copy()
    db_dbscan_cp['cluster'] = model.fit_predict(db_dbscan_cp.to_numpy())
    sns.pairplot(db_dbscan_cp, hue='cluster', diag_kws={"hue": None, "color" : "0.2"})
    plt.savefig('db_dbscan.png')
    plt.clf()




# START OF EXECUTION
if __name__ == '__main__':
    print()
    main()
    print()