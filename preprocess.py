import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import pickle

#from tslearn.clustering import TimeSeriesKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import seaborn as sns

def preprocess(df):
    """Preprocess data for KMeans clustering"""
    
    df_log = np.log1p(df)
    scaler = MinMaxScaler()
    scaler.fit(df_log)
    df_norm = scaler.transform(df_log)
    
    return df_norm

def elbow_plot(df):
    """Create elbow plot from normalized data"""
    
    df_norm = preprocess(df)
    #df_norm = df
    sse = {}
    
    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, random_state=1)
        kmeans.fit(df_norm)
        sse[k] = kmeans.inertia_
    
    plt.title('Elbow plot for K selection')
    plt.xlabel('k')
    plt.ylabel('SSE')
    sns.pointplot(x=list(sse.keys()),
                 y=list(sse.values()))
    plt.show()

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)

    return df[indices_to_keep]#.astype(np.float64)

def main():

    df = pd.read_csv(training_file)

    df = df[['commonname','latitude','longitude']]

    # fig,ax = plt.subplots()
    # for g in np.unique(df['commonname']):
    #     xy = df.loc[df['commonname']==g]
    #     x = xy['longitude']
    #     y = xy['latitude']
    #     ax.scatter(x, y, label = g)
    #     #ax.legend()
    # plt.show()

    df = clean_dataset(df)

    xy = df[['longitude','latitude']]

    k = 170

    kmeans = KMeans(n_clusters=k).fit(xy)


    fig,ax = plt.subplots()
    cm = plt.get_cmap('gist_rainbow')

    ax.set_prop_cycle('color', [cm(1.*i/k) for i in range(k)])

    long = []
    lat = []
    radius = []
    top = []
    numb = []


    i = 0
    for g in np.unique(kmeans.labels_):
        w = np.where(kmeans.labels_==g)
        x = xy['longitude'].iloc[w]
        y = xy['latitude'].iloc[w]
        name = df['commonname'].iloc[w]

        counts = name.value_counts()

        topthree = list(counts.index[0:3])


        ax.scatter(x, y, label = g)

        center = kmeans.cluster_centers_[i]

        w = np.where(kmeans.labels_==i)
        xy_np = xy.iloc[w].to_numpy()

        norm = np.linalg.norm(xy_np - center,axis=1)

        q75, q25 = np.percentile(norm, [75 ,25])
        iqr = q75 - q25
        upper = q75+1.5*iqr
        lower = q25-1.5*iqr

        w2 = np.where(norm >= upper)
        norm = np.delete(norm,w2)
        w2 = np.where(norm <= lower)
        norm = np.delete(norm,w2)

        max = np.amax(norm)
        ax.add_patch(plt.Circle((center[0], center[1]), max, fill = False))
        # ax.scatter(center[0],center[1],c='black')

        long.append(center[0])
        lat.append(center[1])
        numb.append(i)
        # radius.append(max)
        # top.append(topthree)

        i += 1
    ax.plot()
    plt.show()

    #d = {'longitude': long, 'latitude': lat,'radius':radius, 'top three':top}
    d2 = {'longitude':long,'latitude': lat}
    #df2 = pd.DataFrame(data=d)
    df3 = pd.DataFrame(data=d2)

    #df2.to_csv("buffers.csv")
    df3.to_csv("buffers2.csv")

    pass


if __name__ == "__main__":
    training_file = 'summer_points.csv'
    main()