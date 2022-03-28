# This is a sample Python script.


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import seaborn as sns
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier

sns.set()
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def read__credit_card_csv():
    return pd.read_csv('data/creditcard.csv')


def process():
    df = read__credit_card_csv()

    std_scaler = StandardScaler()
    rob_scaler = RobustScaler()

    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    X = df.drop('Class', axis=1)
    y = df['Class']


    # print("Hokins for Unblanced Data- {}", hopkins(X, 150))

    kmean_exp(X, y)
    start = time.time()
    kmean_exp_sillhoute(X, y)
    print("K means without feature reduction {}".format(time.time() - start))
    silhouette_coefficent(X,y)
    kmean_cluster(X, y)
    plot_confusion_kmeans(X, y)

    expectation_maximization(X, y)
    kmean_exp_sillhoute(X,y)

    pca_chose_dimensions(X, y)
    pca_chose_dimension_plot_eigenvalue_distribution(X,y)
    ica_chose_dimensions(X, y)
    random_projection(X,y)
    random_projection_recontction(X,y)
    random_projection_recontction_variation(X,y)
    feature_selection(X, y)
    kmean_exp(pca_chose_dimensions_clustering(X,y,20), y, file="elbow_curve_kmeans_after_dim_red.png")

    # Kmean Clustering after PCA reduction
    X_red = pca_chose_dimensions_clustering(X, y, 20)
    start = time.time()
    kmean_exp_sillhoute(X_red,y , file="sill_houte_kmeans_after_pca_dim_red.png")
    print("K means with PCA feature reduction {}".format(time.time() - start))

    start = time.time()
    silhouette_coefficent(X_red,y,file="k_means_sillhouette_coefficent_after_pca_reduction_{}.png")
    print("K means Sillhoute coefficent with PCA feature reduction {}".format(time.time() - start))

    # Kmeans Clustering after ICA reduction
    start = time.time()
    X = X.to_numpy()
    y = y.to_numpy()
    X_ica = ica_chose_dimensions_reduction(X, y, 1)
    kmean_exp(X_ica, y, file="elbow_curve_kmeans_after_dim_red_ica.png")
    print("K means Elbow with ICA feature reduction {}".format(time.time() - start))
    start = time.time()
    kmean_exp_sillhoute(X_ica, y, file="sill_houte_score_kmeans_after_ica_dim_red.png")
    print("K means Sillhoute Score with ICA feature reduction {}".format(time.time() - start))

    start = time.time()
    silhouette_coefficent(X_ica, y, file="sill_houte_coeffiecnt_kmeans_after_ica_dim_red_{}.png")
    print("K means Sillhoute coefficent with ICA feature reduction {}".format(time.time() - start))

    # Kmeans Clustering after random projetion reduction

    #Kmeans clustering Random Projection
    X_proj = random_project_reconstruction_with_num_of_cluster(X,y, ncluster=16)
    kmean_exp(X_proj, y, file="elbow_curve_kmeans_after_dim_red_random_proj.png")
    print("K means Elbow with Randonm Projection feature reduction {}".format(time.time() - start))
    start = time.time()
    kmean_exp_sillhoute(X_proj, y, file="sill_houte_score_kmeans_after_random_proj_dim_red.png")
    print("K means Sillhoute Score with Randonm Projection feature reduction {}".format(time.time() - start))

    start = time.time()
    silhouette_coefficent(X_proj, y, file="sill_houte_coeffiecnt_kmeans_after_random_proj_dim_red_{}.png")
    print("K means Sillhoute coefficent with Randonm Projection feature reduction {}".format(time.time() - start))



    #Kmeans clustering for using feature selection
    X_feature = feature_selection_sort_return(X, y)
    kmean_exp(X_feature, y, file="elbow_curve_kmeans_after_dim_red_feature_proj.png")
    print("K means Elbow with Feature Selection Projection feature reduction {}".format(time.time() - start))
    start = time.time()
    kmean_exp_sillhoute(X_feature, y, file="sill_houte_score_kmeans_after_feature_selection_dim_red.png")
    print("K means Sillhoute Score with Feature Selection feature reduction {}".format(time.time() - start))

    start = time.time()
    silhouette_coefficent(X_feature, y, file="sill_houte_coeffiecnt_kmeans_after_feature_selection_proj_dim_red_{}.png")
    print("K means Sillhoute coefficent with Feature Projection feature reduction {}".format(time.time() - start))




    #Expectation Maximization
    # Expectation Maximization Clustering after PCA reduction
    X_red = pca_chose_dimensions_clustering(X, y, 20)
    start = time.time()
    expectation_maximization(X_red,y , file="expectation_maximization_after_pca_dim_red.png")
    print("K means with Expecatation Maximization feature reduction {}".format(time.time() - start))



    # Expectation Maximization Clustering after ICA reduction
    X_ica = ica_chose_dimensions_reduction(X, y, 1)
    expectation_maximization(X_ica,y , file="expectation_maximization_after_ica_dim_red.png")
    print("Expectation Maximization Elbow with ICA feature reduction {}".format(time.time() - start))


    # Expectation Maximization Clustering after random projetion reduction
    X_proj = random_project_reconstruction_with_num_of_cluster(X,y, ncluster=16)
    expectation_maximization(X_proj, y, file="expectation_maximization_after_random_proj_dim_red.png")
    print("Expectation Maximization Elbow with Randonm Projection feature reduction {}".format(time.time() - start))




    #Expectation Maximization clustering for using feature selection
    X_feature = feature_selection_sort_return(X, y)
    expectation_maximization(X_feature, y, file="elbow_curve_expectation_maximization_after_dim_red_feature_proj.png")
    print("Expectation Maximization Sillhoute coefficent with Feature Projection feature reduction {}".format(time.time() - start))


    #Neural Network
    #Kmeans
    # amount of fraud classes 492 rows.
    fraud_df = df.loc[df['Class'] == 1]
    non_fraud_df = df.loc[df['Class'] == 0][:len(fraud_df)]
    #
    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
    #
    X = normal_distributed_df.drop('Class', axis=1).to_numpy()
    y = normal_distributed_df['Class'].to_numpy()
    #
    cluster_label = kmeans_cluster_label(X, y)
    X_y_with_cluster = np.concatenate((X, cluster_label.reshape(len(cluster_label), 1), y.reshape(len(cluster_label), 1)), axis=1)
    neural_network_size(X_y_with_cluster, file='neural_network_size_after_clustering')
    neural_network_size_time(X_y_with_cluster, file='neural_network_size_after_clustering_time_pca')
    X_y = np.concatenate(
        (X, y.reshape(len(cluster_label), 1)), axis=1)
    neural_network_size_time(X_y,file='neural_network_size_orig_time')


    #Expectation Maximization
    cluster_label = expectation_maximization_cluster_label(X, y)
    X_y_with_cluster = np.concatenate(
        (X, cluster_label.reshape(len(cluster_label), 1), y.reshape(len(cluster_label), 1)), axis=1)
    neural_network_size(X_y_with_cluster, file='neural_network_size_after_clustering_expectation')









def neural_network_size(new_df, file='neural_network_size_after_clustering'):
    f, (ax1) = plt.subplots(1, 1, figsize=(20, 6))
    sizes = list()
    traiing_scores = list()
    cross_val_score = list()
    for size in range(50, len(new_df), 50):
        sizes.append(size)
        shuffled = np.random.shuffle(new_df)
        X, y = new_df[:, :-1], new_df[:, -1]
        clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(7, 7, 7), random_state=1, activation='relu',
                                learning_rate='adaptive', max_iter=100000)

        scr = cross_validate(clf, X, y, cv=5, return_train_score=True, n_jobs=-1)
        traiing_scores.append(scr['train_score'].mean())
        cross_val_score.append(scr['test_score'].mean())
    plt_df = pd.DataFrame(
            {'x_values': sizes, 'training score': traiing_scores, 'Cross Validation Score': cross_val_score})

    plt.plot('x_values', 'training score', data=plt_df, marker='o', markerfacecolor='blue', markersize=12,
                 color='skyblue',
                 linewidth=4)
    plt.plot('x_values', 'Cross Validation Score', data=plt_df, marker='o', markerfacecolor='green', markersize=12,
                 color='lightgreen',
                 linewidth=4)

        # show legend
    plt.legend()
    plt.savefig('{}.png'.format(file))




def neural_network_size_time(new_df, file='neural_network_size_after_clustering'):
    f, (ax1) = plt.subplots(1, 1, figsize=(20, 6))
    sizes = list()
    traiing_scores = list()
    cross_val_score = list()
    exection_time = list()
    for size in range(50, len(new_df), 50):
        sizes.append(size)
        shuffled = np.random.shuffle(new_df)
        X, y = new_df[:, :-1], new_df[:, -1]
        start = time.time()
        clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(7, 7, 7), random_state=1, activation='relu',
                                learning_rate='adaptive', max_iter=100000)

        scr = cross_validate(clf, X, y, cv=5, return_train_score=True, n_jobs=-1)
        exection_time.append(time.time() - start)
        # traiing_scores.append(scr['train_score'].mean())
        # cross_val_score.append(scr['test_score'].mean())
    plt_df = pd.DataFrame(
            {'x_values': sizes, 'Execution Time': exection_time})

    plt.plot('x_values', 'Execution Time', data=plt_df, marker='o', markerfacecolor='blue', markersize=12,
                 color='skyblue',
                 linewidth=4)

        # show legend
    plt.legend()
    plt.savefig('{}.png'.format(file))








k = 4
n_draws = 500
sigma = .7
random_state = 0
dot_size = 50
cmap = 'viridis'
from sklearn.metrics import mutual_info_score
from sklearn.random_projection import SparseRandomProjection
import numpy as np
import time

from xgboost import XGBClassifier
from matplotlib import pyplot



def feature_selection_sort_return(X,y):
    model = XGBClassifier()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    return X[:,(-importance).argsort()[:4]];
    # summarize feature importance
    # for i, v in enumerate(importance):
    #     print()


def feature_selection(X, y):
    fig, ax = plt.subplots(figsize=(11, 7))
    model = XGBClassifier()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance

    pyplot.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='center')
    pyplot.bar([x for x in X.columns], importance)

    pyplot.savefig("feature_selection_XGB.png")


def random_projection(X, y):
    time_random = []
    time_PCA = []
    for i in range(1, 30, 3):
        start = time.time()
        transformer = SparseRandomProjection(n_components=i)
        X_new = transformer.fit(X)
        time_random.append(time.time() - start)

        start = time.time()
        pca = PCA(n_components=i)
        pca.fit(X)
        time_PCA.append(time.time() - start)

    plt_df = pd.DataFrame(
        {'x_values': range(1, 30, 3), 'Random Projection': time_random, 'PCA': time_PCA})

    plt.plot('x_values', 'Random Projection', data=plt_df, marker='o', markerfacecolor='blue', markersize=12,
             color='skyblue',
             linewidth=4)
    plt.plot('x_values', 'PCA', data=plt_df, marker='o', markerfacecolor='green', markersize=12,
             color='lightgreen',
             linewidth=4)

    # show legend
    plt.legend()
    plt.savefig('{}.png'.format("runtime_pca_random.png"))


def random_project_reconstruction_with_num_of_cluster(X, y, ncluster=16):
    transformer = SparseRandomProjection(n_components=ncluster)
    return transformer.fit_transform(X)



def random_projection_recontction_variation(X, y):
    reconstruction = []
    print("Reconstruction error for randome projection")
    X = X.to_numpy()
    for i in range(1, 30, 3):
        transformer = SparseRandomProjection(n_components=16)
        X_new = transformer.fit_transform(X)
        X_reconstructed = inverse_transform_rp(transformer, X_new, X)
        reconstruction.append(((X - X_reconstructed) ** 2).mean())

    fig, ax = plt.subplots(figsize=(9, 7))
    plt_df = pd.DataFrame(
        {'x_values': range(1, 30, 3), 'Reconstruction Error': reconstruction})

    plt.plot('x_values', 'Reconstruction Error', data=plt_df, marker='o', markerfacecolor='blue', markersize=12,
             color='skyblue',
             linewidth=4)

    # show legend
    plt.legend()
    plt.savefig('{}.png'.format("reconstruction_error_random_proj_varaition"))
    print("Finished Reconstruction error for randome projection")


def random_projection_recontction(X, y):
    reconstruction = []
    print("Reconstruction error for randome projection")
    X = X.to_numpy()
    for i in range(1, 30, 3):
        transformer = SparseRandomProjection(n_components=i)
        X_new = transformer.fit_transform(X)
        X_reconstructed = inverse_transform_rp(transformer, X_new, X)
        reconstruction.append(((X - X_reconstructed) ** 2).mean())

    fig, ax = plt.subplots(figsize=(9, 7))
    plt_df = pd.DataFrame(
        {'x_values': range(1, 30, 3), 'Reconstruction Error': reconstruction})

    plt.plot('x_values', 'Reconstruction Error', data=plt_df, marker='o', markerfacecolor='blue', markersize=12,
             color='skyblue',
             linewidth=4)

    # show legend
    plt.legend()
    plt.savefig('{}.png'.format("reconstruction_error_random_proj"))
    print("Finished Reconstruction error for randome projection")


def inverse_transform_rp(rp, X_transformed, X_train):
    return X_transformed.dot(rp.components_.toarray()) + np.mean(X_train, axis=0)


def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def ica_chose_dimensions_reduction(X, y, components):
    ica = FastICA(random_state=5)
    ica.set_params(n_components=components)
    return ica.fit_transform(X)


def ica_chose_dimensions(X, y):
    dims = list(np.arange(1, X.shape[1]))
    dims.append(X.shape[1])
    ica = FastICA(random_state=5)
    kurt = []

    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt.append(tmp.abs().mean())

    plt.figure()
    plt.title("ICA Kurtosis: ")
    plt.xlabel("Independent Components")
    plt.ylabel("Avg Kurtosis Across IC")
    plt.plot(dims, kurt, 'b-')
    plt.grid(False)
    plt.savefig("ica_kurtosisis.png")
import plotly.express as px

def pca_chose_dimensions_clustering(X, y, ncomponent):
    pca = PCA(n_components=ncomponent)
    return pca.fit_transform(X, y)

def pca_chose_dimension_plot_eigenvalue_distribution(X, y):
    pca = PCA()
    components = pca.fit_transform(X)
    plt.hist(pca.explained_variance_)
    plt.savefig('eigen_value_distribution.png')


def pca_chose_dimensions(X, y):
    pca = PCA()
    pca.fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    plt.title("Explained Variance vs Dimensions")
    plt.xlabel("Dimensions")
    plt.ylabel("Explained Variance")
    plt.plot(range(0, len(cumsum)), cumsum)
    plt.savefig("pca_variance_dimesnion.png")
    # d = np.argmax(cumsum >= 0.95) + 1


def plot_confusion_kmeans(X, y):
    fig, ax = plt.subplots(figsize=(9, 7))
    km = KMeans(n_clusters=2,
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0)
    y_km = km.fit_predict(X)

    cm = pair_confusion_matrix(y, y_km)
    data = {'y_Actual': y,
            'y_Predicted': y_km
            }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    sn.heatmap(confusion_matrix, annot=True)
    plt.savefig("kmeans_confusion_matrix.png")


def kmean_cluster_visulaize(X, nclusters, file="kmeans_cluster_visualize.png"):
    # Initialize the class object
    kmeans = KMeans(n_clusters=nclusters)

    # predict the labels of clusters.
    label = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    u_labels = np.unique(label)

    # plotting the results:

    for i in u_labels:
        plt.scatter(X[label == i, 0], X[label == i, 0], label=i)

    # plt.scatter(centroids[:, 0], centroids[:, 0], s=80, color='k')
    plt.legend()
    plt.savefig(file)


def kmean_exp(X, y, file="elbow_curve_kmeans.png"):
    Sum_of_squared_distances = []
    fig, ax = plt.subplots(figsize=(9, 7))
    K = range(1, 10)
    for num_clusters in K:
        print("Running K means for {}", num_clusters)
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)
        Sum_of_squared_distances.append(kmeans.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Sum of squared distances / Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.savefig(file)

    # kmeans = KMeans(n_clusters=4)
    # kmeans.fit(X)
    # y_kmeans = kmeans.predict(X)
    #
    # # visualize prediction
    # fig, ax = plt.subplots(figsize=(9, 7))
    # ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=dot_size, cmap=cmap)
    #
    # # get centers for plot
    # centers = kmeans.cluster_centers_
    # ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75)
    # plt.title('sklearn k-means', fontsize=18, fontweight='demi')
    # plt.savefig("kmeans_cluster.png")

    # kmeans = KMeans(n_clusters=2)
    # kmeans.fit(X)
    # print(np.sum(kmeans.labels_ == y)/ len(kmeans.labels_))


def kmean_exp_sillhoute(X, y, file="sill_houte_kmeans.png"):
    Sill_score = []
    fig, ax = plt.subplots(figsize=(9, 7))
    K = range(2, 10)
    for num_clusters in K:
        print("Running K means for {}", num_clusters)
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)
        labels = kmeans.labels_
        Sill_score.append(silhouette_score(X, labels, metric='euclidean'))
    plt.plot(K, Sill_score, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score vs Number of Clusters')
    plt.savefig(file)



def kmeans_cluster_label(X, y):
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    return kmeans.predict(X)


def kmean_cluster(X, y):
    X = X.to_numpy()
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    # visualize prediction
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=dot_size, cmap=cmap)

    # get centers for plot
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75)
    plt.title('sklearn k-means', fontsize=18, fontweight='demi')
    plt.savefig("kmeans_cluster.png")


def silhouette_coefficent(X, y, file="k_means_sillhouette_coefficent_{}.png"):
    for l in range(8, 9):
        print("Sillhoutte Coefficent - {}".format(l))
        fig, ax = plt.subplots(figsize=(9, 7))
        km = KMeans(n_clusters=l,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    tol=1e-04,
                    random_state=0)
        y_km = km.fit_predict(X)
        print(km.cluster_centers_)
        import numpy as np
        from matplotlib import cm
        from sklearn.metrics import silhouette_samples
        cluster_labels = np.unique(y_km)
        n_clusters = cluster_labels.shape[0]
        silhouette_vals = silhouette_samples(X,
                                             y_km,
                                             metric='euclidean')
        y_ax_lower, y_ax_upper = 0, 0
        yticks = []
        for i, c in enumerate(cluster_labels):
            c_silhouette_vals = silhouette_vals[y_km == c]
            c_silhouette_vals.sort()

            y_ax_upper += len(c_silhouette_vals)
            color = cm.jet(float(i) / n_clusters)
            plt.barh(range(y_ax_lower, y_ax_upper),
                     c_silhouette_vals,
                     height=1.0,
                     edgecolor='none',
                     color=color)
            yticks.append((y_ax_lower + y_ax_upper) / 2.)
            y_ax_lower += len(c_silhouette_vals)

        silhouette_avg = np.mean(silhouette_vals)
        plt.axvline(silhouette_avg,
                    color="red",
                    linestyle="--")
        plt.yticks(yticks, cluster_labels + 1)
        plt.ylabel('Cluster')
        plt.xlabel('Silhouette coefficient')
        plt.savefig(file.format(l))


# def expectation_maximization(X,y):
#     gm = GaussianMixture(n_components=2, random_state=0).fit(X)

def SelBest(arr: list, X: int) -> list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx = np.argsort(arr)[:X]
    return arr[dx]


def expectation_maximization_cluster_label(X, y):
    gmm = GaussianMixture(8, n_init=10).fit(X)
    return gmm.predict(X)


def expectation_maximization(X, y, file="expectation_maximization_num_of_clusters.png"):
    n_clusters = np.arange(2, 10)
    bics = []
    aics = []
    bics_err = []
    aics_err = []
    iterations = 20
    fig, ax = plt.subplots(figsize=(9, 7))
    for n in n_clusters:
        print("Trying Cluster - {}", n)

        gmm = GaussianMixture(n, n_init=10).fit(X)

        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))

    plt.errorbar(n_clusters, bics, label='BIC')
    plt.errorbar(n_clusters, aics, label='BIC')
    plt.title("BIC/AIC Scores", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("No. of clusters")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(file)


if __name__ == '__main__':
    process()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
