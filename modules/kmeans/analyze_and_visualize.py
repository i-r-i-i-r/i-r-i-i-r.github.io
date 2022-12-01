import altair as alt
from sklearn.cluster import KMeans

_BRUSH = alt.selection_interval(name='brush') # selection of type "interval"

def get_clusters(n_clusters, penguins, cols):
    kmeans = KMeans(n_clusters=n_clusters)
    est = kmeans.fit(penguins[cols].values)
    df = penguins.copy()
    df['labels'] = est.labels_.astype('str')
    return df

def get_plot(x, y, df):
    centers = df.groupby('labels').mean()
    return (alt.Chart(df)
        .mark_point(size=100)
        .encode(
            x=alt.X(x, scale=alt.Scale(zero=False)),
            y=alt.Y(y, scale=alt.Scale(zero=False)),
            shape='labels',
            color='species'
        ).add_selection(_BRUSH).properties(width=800) +
        alt.Chart(centers)
            .mark_point(size=250, shape='cross', color='black')
            .encode(x=x+':Q', y=y+':Q')
    )



