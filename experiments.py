from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from scipy import linalg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import ShuffleSplit, train_test_split, learning_curve, validation_curve, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import seaborn as sns
from yellowbrick.cluster import SilhouetteVisualizer

class experiments:
    def __init__(self):
        self.random_seed = 910117883
        self.output_file = 'output.txt'
        self.scoring = "accuracy"
        self.datasets = {"Heart Data": None, "Weather Data": None}
        self.clusters = range(2, 6)
        self.train_sizes = np.linspace(0.01, 1.0, 20)
        self.cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=self.random_seed)
        self.nn_model = MLPClassifier(max_iter=1000, random_state=self.random_seed)
        self.nn_hyperparameters = {
           "hidden_layer_sizes": [(10,), (20,), (10, 10,), (20, 20,), (10, 10, 10), (20, 20, 20)],
            "alpha": 10.0**-np.arange(-1, 5)
        }
        self.tuned_nn = None

    def write_to_output(self, text):
        with open(self.output_file, 'a') as f:
            f.write(text)
            f.write('\n')

    def prep_heart_data(self):
        heart_data = pd.read_csv('datasets/heart_2020_cleaned.csv')
        heart_data = heart_data.dropna()
        # Map binary columns to 0 or 1
        binary_columns = ["HeartDisease", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "PhysicalActivity",
                          "Asthma", "KidneyDisease", "SkinCancer"]
        for column in binary_columns:
            heart_data[column] = heart_data[column].map({'Yes': 1, 'No': 0})

        # Balance heart data, which has significantly more False cases than True
        negative_cases = heart_data[heart_data.HeartDisease == 0].index
        sample_size = heart_data[heart_data.HeartDisease == 1].shape[0]
        negative_sample = np.random.choice(negative_cases, sample_size, replace=False)
        balanced_heart_data = pd.DataFrame()
        balanced_heart_data = balanced_heart_data.append(heart_data[heart_data.HeartDisease == 1])
        balanced_heart_data = balanced_heart_data.append(heart_data.iloc[negative_sample, :])

        # Mix the sample and drop the index, sample only 50% to keep run-time manageable
        balanced_heart_data = balanced_heart_data.sample(frac=1).reset_index()
        balanced_heart_data = balanced_heart_data.drop("index", axis=1)

        # Separate x and y
        y_vals = balanced_heart_data.HeartDisease
        x_vals = balanced_heart_data.drop("HeartDisease", axis=1)

        # Encode textual columns to numerical values
        textual_columns = ['Sex', 'AgeCategory', 'Race', 'Diabetic', 'GenHealth']
        column_transformer = make_column_transformer((OrdinalEncoder(), textual_columns), remainder='passthrough')
        x_vals = column_transformer.fit_transform(x_vals)
        x_vals = pd.DataFrame(data=x_vals, columns=column_transformer.get_feature_names_out())
        for column in x_vals.columns:
            x_vals = x_vals.rename(columns={column: column.split('__')[-1]})

        self.datasets["Heart Data"] = {"x": x_vals, "y": y_vals}

    def prep_weather_data(self):
        weather_data = pd.read_csv('datasets/weatherAUS.csv')
        weather_data = weather_data.dropna()

        # Map binary columns to 0 or 1
        binary_columns = ["RainToday", "RainTomorrow"]
        for column in binary_columns:
            weather_data[column] = weather_data[column].map({'Yes': 1, 'No': 0})

        # Extract month from date column and then drop the date
        weather_data["Date"] = pd.to_datetime(weather_data["Date"])
        weather_data["Month"] = weather_data["Date"].dt.month
        weather_data = weather_data.drop("Date", axis=1)

        # Mix up the sample and drop the index, sample 50% to keep run-time manageable
        weather_data = weather_data.sample(frac=1).reset_index()
        weather_data = weather_data.drop("index", axis=1)

        # Separate x and y values
        y_vals = weather_data.RainTomorrow
        x_vals = weather_data.drop("RainTomorrow", axis=1)

        # Drop irrelevant columns. We are predicting weather tomorrow, so 3 pm should be most relevant
        drop_columns = ['WindDir9am', 'WindDir3pm', 'WindGustDir', 'WindSpeed9am', 'Humidity9am', 'Pressure9am',
                        'Cloud9am', 'Temp9am', 'Location']
        for col in drop_columns:
            x_vals = x_vals.drop(col, axis=1)

        self.datasets["Weather Data"] = {"x": x_vals, "y": y_vals}

    def plot_and_fill_between(self, axis, x, y_mean, y_std, color, label):
        axis.plot(x, y_mean, "o-", color=color, label=label)
        axis.fill_between(
            x,
            y_mean - y_std,
            y_mean + y_std,
            alpha=0.1,
            color=color
        )

    def plot_learning_curve(self, axis, train_sizes, train_scores, cv_scores):
        mean_train_score = np.mean(train_scores, axis=1)
        std_train_score = np.std(train_scores, axis=1)
        mean_cv_score = np.mean(cv_scores, axis=1)
        std_cv_score = np.std(cv_scores, axis=1)
        self.plot_and_fill_between(axis, train_sizes, mean_train_score, std_train_score, "r", f"Training {self.scoring}")
        self.plot_and_fill_between(axis, train_sizes, mean_cv_score, std_cv_score, "g", f"Cross-validation {self.scoring}")
        axis.legend(loc="best")
        axis.grid()

    def plot_fit_time(self, axis, train_sizes, fit_times):
        mean_fit_time = np.mean(fit_times, axis=1)
        std_fit_time = np.std(fit_times, axis=1)
        self.plot_and_fill_between(axis, train_sizes, mean_fit_time, std_fit_time, "r", "Training fit time")
        axis.set_xlabel("Number of training examples")
        axis.set_ylabel("Fit time")
        axis.set_title("Model scalability")
        axis.grid()

    def plot_score_time(self, axis, train_sizes, score_times):
        mean_score_time = np.mean(score_times, axis=1)
        std_score_time = np.std(score_times, axis=1)
        self.plot_and_fill_between(axis, train_sizes, mean_score_time, std_score_time, "r", "Cross-validation score time")
        axis.set_xlabel("Number of training examples")
        axis.set_ylabel("Score time")
        axis.set_title("Model scalability")
        axis.grid()

    def plot_performance_train(self, axis, fit_times, cv_scores):
        mean_fit_time = np.mean(fit_times, axis=1)
        mean_test_score = np.mean(cv_scores, axis=1)
        std_test_score = np.std(cv_scores, axis=1)

        fit_time_sort = mean_fit_time.argsort()
        sorted_mean_fit_time = mean_fit_time[fit_time_sort]
        sorted_mean_test = mean_test_score[fit_time_sort]
        sorted_std_test = std_test_score[fit_time_sort]

        self.plot_and_fill_between(axis, sorted_mean_fit_time, sorted_mean_test, sorted_std_test, "g", "Model performance")
        axis.set_xlabel("Fit time")
        axis.set_ylabel(f"{self.scoring}")
        axis.set_title("Model performance")
        axis.grid()

    def plot_performance_test(self, axis, score_times, cv_scores):
        mean_score_time = np.mean(score_times, axis=1)
        mean_test_score = np.mean(cv_scores, axis=1)
        std_test_score = np.std(cv_scores, axis=1)

        score_time_sort = mean_score_time.argsort()
        sorted_mean_score_time = mean_score_time[score_time_sort]
        sorted_mean_test = mean_test_score[score_time_sort]
        sorted_std_test = std_test_score[score_time_sort]

        self.plot_and_fill_between(axis, sorted_mean_score_time, sorted_mean_test, sorted_std_test, "g", "Model performance")
        axis.set_xlabel("Score time")
        axis.set_ylabel(f"{self.scoring}")
        axis.set_title("Model performance")
        axis.grid()

    def generate_learning_curves(self, x_train, y_train, model, title):
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].set_title(title)
        axes[0].set_xlabel("Number of training examples")
        axes[0].set_ylabel(f"{self.scoring}")

        train_sizes, train_scores, cv_scores, fit_times, score_times = learning_curve(
            model,
            x_train,
            y_train,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=-1,
            train_sizes=self.train_sizes,
            return_times=True,
        )

        self.plot_learning_curve(axes[0], train_sizes, train_scores, cv_scores)
        if "K-Nearest Neighbors" in title:
            self.plot_score_time(axes[1], train_sizes, score_times)
            self.plot_performance_test(axes[2], score_times, cv_scores)
        else:
            self.plot_fit_time(axes[1], train_sizes, fit_times)
            self.plot_performance_train(axes[2], fit_times, cv_scores)

        file_name = title.replace(" ", "")
        plt.savefig(f"images/{file_name}.png")
        plt.close()

    def plot_validation_curve(self, train_scores, cv_scores, title, param_name, param_range, xscale):
        mean_train_score = np.mean(train_scores, axis=1)
        std_train_score = np.std(train_scores, axis=1)
        mean_cv_score = np.mean(cv_scores, axis=1)
        std_cv_score = np.std(cv_scores, axis=1)

        plt.title(title)
        plt.xlabel(param_name)
        plt.ylabel(f"{self.scoring}")
        plt.grid()
        plt.xscale(xscale)
        if not isinstance(param_range[0], str):
            plt.xticks(param_range)

        self.plot_and_fill_between(plt, param_range, mean_train_score, std_train_score, "r", f"Training {self.scoring}")
        self.plot_and_fill_between(plt, param_range, mean_cv_score, std_cv_score, "g", f"Cross-validation {self.scoring}")
        plt.legend(loc="best")

        file_name = title.replace(" ", "") + "_" + param_name
        plt.savefig(f"images/{file_name}.png")
        plt.close()

    def generate_validation_curves(self, x_train, y_train, model, hyperparams, title):
        for param in hyperparams:
            param_range = hyperparams[param]
            train_scores, cv_scores = validation_curve(
                model,
                x_train,
                y_train,
                cv=self.cv,
                param_name=param,
                param_range=param_range,
                scoring=self.scoring,
                n_jobs=-1
            )
            xscale = "linear"
            if param == "hidden_layer_sizes":
                param_range = ["(10)", "(20)", "(10, 10)", "(20, 20)", "(10, 10, 10)", "(20, 20, 20)"]
            if param == "alpha":
                xscale = "log"
            self.plot_validation_curve(train_scores, cv_scores, title, param, param_range, xscale)

    def generate_loss_curve(self, x_train, y_train, model, title):
        model.fit(x_train, y_train)
        loss_curve = model.loss_curve_
        plt.plot(loss_curve, "o-", color="g", label="loss curve")
        plt.title(title)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.grid()
        file_name = title.replace(" ", "")
        plt.savefig(f"images/{file_name}.png")
        plt.close()

    def get_test_performance(self, x_train, y_train, x_test, y_test, model):
        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)
        self.write_to_output(f"Train accuracy: {train_score} | Test accuracy: {test_score}")

    def tune_hyperparams(self, x_train, y_train, model, hyperparams, model_name, data_set):
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=hyperparams,
                                   n_jobs=-1,
                                   cv=self.cv,
                                   scoring=self.scoring)
        result = grid_search.fit(x_train, y_train)

        self.write_to_output(f"Tuned hyperparams for: {model_name} on dataset: {data_set} \n Params: {result.best_estimator_.get_params()}")
        self.write_to_output(f"Fit time: {np.average(result.cv_results_['mean_fit_time'])}")
        self.write_to_output(f"Score time: {np.average(result.cv_results_['mean_score_time'])}")
        return result.best_estimator_

    def neural_net_experiment(self):
        for data_set in self.datasets:
            self.write_to_output(f"NEURAL NETWORK FOR {data_set} \n ___________________________________________________")
            vc_title = f"Neural Network Validation Curve for {data_set}"
            lc_title = f"Neural Network Learning Curve for {data_set}"
            loss_title = f"Neural Network Loss Curve for {data_set}"

            x = self.datasets[data_set]["x"]
            y = self.datasets[data_set]["y"]

            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.2,
                random_state=self.random_seed
            )
            # MLP is sensitive to feature scaling, so it is recommended to scale the data (https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

            self.generate_validation_curves(x_train, y_train, self.nn_model, self.nn_hyperparameters, vc_title)
            self.tuned_nn = self.tune_hyperparams(x_train, y_train, self.nn_model, self.nn_hyperparameters, "Neural Network", data_set)
            self.generate_learning_curves(x_train, y_train, self.tuned_nn, lc_title)
            self.generate_loss_curve(x_train, y_train, self.tuned_nn, loss_title)
            self.get_test_performance(x_train, y_train, x_test, y_test, self.tuned_nn)

    # this method is borrowed from the sklearn example here: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
    def bench_cluster(self, algo, name, n_clusters, data):
        t0 = perf_counter()
        estimator = make_pipeline(StandardScaler(), algo).fit(data)
        fit_time = perf_counter() - t0
        results = [name, n_clusters, fit_time]

        # The silhouette score requires the full dataset
        silhouette = [
            metrics.silhouette_score(
                data,
                estimator.predict(data),
                metric="euclidean",
                sample_size=10000
            )
        ]
        results += silhouette

        # Show the results
        formatter_result = (
            "{:9s}\t{:.0f}\t{:.3f}s\t{:.3f}"
        )
        self.write_to_output(formatter_result.format(*results))
        return pd.DataFrame(data={'clusters': [n_clusters], 'fit_time': [fit_time], 'silhouette': silhouette})

    def plot_cluster_results(self, data_set, algo, results):
        title = 'Silhouette Scores for ' + data_set
        plt.title(title)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Scores')
        plt.plot(results.clusters, results.silhouette, label=f'{algo} Silhouette Score')
        plt.grid()
        plt.legend(loc="best")

    def run_pca(self, x, data_set):
        pca = PCA(random_state=self.random_seed)
        x_pca = pca.fit_transform(x)
        exp_var = pca.explained_variance_ratio_
        eigen = np.cumsum(exp_var)
        plt.figure("PCA")
        plt.bar(range(0, len(exp_var)), exp_var, alpha=0.5, align='center',
                label='Individual explained variance')
        plt.step(range(0, len(eigen)), eigen, where='mid',
                 label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.title(f"PCA on {data_set}")
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'images/{data_set}_PCA_EV.png')
        plt.close()

    def run_ica(self, x, data_set):
        plt.figure("ICA")
        n_components = range(1, x.shape[1] + 1)
        kurtosis = pd.DataFrame(index=n_components, columns=['kurtosis'])
        for n in n_components:
            ica = FastICA(n_components=n, random_state=self.random_seed)
            k = pd.DataFrame(ica.fit_transform(x)).kurtosis().abs().mean()
            kurtosis.loc[n, 'kurtosis'] = k
        plt.plot(kurtosis, label=f"{data_set} kurtosis")

    def run_rca(self, x, data_set):
        plt.figure("RCA")
        n_components = range(1, x.shape[1] + 1)
        kurtosis = pd.DataFrame(index=n_components, columns=['kurtosis'])
        for n in n_components:
            rca = SparseRandomProjection(n_components=n, random_state=self.random_seed)
            k = pd.DataFrame(rca.fit_transform(x)).kurtosis().abs().mean()
            kurtosis.loc[n, 'kurtosis'] = k
        plt.plot(kurtosis, label=f"{data_set} Kurtosis")

    def run_lda(self, x, y, data_set):
        fig = f"LDA-{data_set}"
        if data_set == 'Heart Data':
            i1 = 1
            i2 = 5
        else:
            i1 = 0
            i2 = 1
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=self.random_seed
        )
        knn = KNeighborsClassifier(n_neighbors=3)
        lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
        lda.fit(x_train, y_train)
        knn.fit(lda.transform(x_train), y_train)
        acc_knn = knn.score(lda.transform(x_test), y_test)
        y_pred = lda.fit(x, y).predict(x)

        plt.figure(fig)
        self.plot_lda(fig, lda, x, y, y_pred, i1, i2)
        self.plot_ellipse(fig, lda.means_[0], lda.covariance_, "red", i1, i2)
        self.plot_ellipse(fig, lda.means_[1], lda.covariance_, "blue", i1, i2)
        plt.title("LDA, KNN (k={})\nTest accuracy = {:.2f}".format(3, acc_knn))
        plt.savefig(f'images/{data_set}_LDA_KNN.png')
        plt.close()

    def plot_lda(self, fig, lda, X, y, y_pred, i1, i2):
        plt.figure(fig)

        tp = y == y_pred  # True Positive
        tp0, tp1 = tp[y == 0], tp[y == 1]
        X0, X1 = X[y == 0], X[y == 1]
        X0_tp, X0_fp = X0[tp0], X0[~tp0]
        X1_tp, X1_fp = X1[tp1], X1[~tp1]

        # class 0: dots
        plt.scatter(X0_tp[:, i1], X0_tp[:, i2], marker=".", color="red")
        plt.scatter(X0_fp[:, i1], X0_fp[:, i2], marker="x", s=20, color="#990000")  # dark red

        # class 1: dots
        plt.scatter(X1_tp[:, i1], X1_tp[:, i2], marker=".", color="blue")
        plt.scatter(
            X1_fp[:, i1], X1_fp[:, i2], marker="x", s=20, color="#000099"
        )  # dark blue

        # means
        plt.plot(
            lda.means_[0][i1],
            lda.means_[0][i2],
            "*",
            color="yellow",
            markersize=15,
            markeredgecolor="grey",
        )
        plt.plot(
            lda.means_[1][i1],
            lda.means_[1][i2],
            "*",
            color="yellow",
            markersize=15,
            markeredgecolor="grey",
        )

    def plot_ellipse(self, fig, mean, cov, color, i1, i2):
        plt.figure(fig)
        splot = plt.subplot()
        v, w = linalg.eigh(cov)
        u = w[i1] / linalg.norm(w[i1])
        angle = np.arctan(u[i2] / u[i1])
        angle = 180 * angle / np.pi  # convert to degrees
        # filled Gaussian at 2 standard deviation
        ell = mpl.patches.Ellipse(
            mean,
            2 * v[i1] ** 0.5,
            2 * v[i2] ** 0.5,
            180 + angle,
            facecolor=color,
            edgecolor="black",
            linewidth=2,
        )
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.2)
        splot.add_artist(ell)
        splot.set_xticks(())
        splot.set_yticks(())

    def kurtosis_plot(self, algorithm, part):
        plt.figure(algorithm)
        plt.ylabel('Kurtosis')
        plt.xlabel('n_components')
        plt.title(f"Average kurtosis for n_components for {algorithm}")
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'images/part{part}_{algorithm}_kurtosis.png')
        plt.close()

    def part2_experiment(self):
        for data_set in self.datasets:
            self.write_to_output(f"PART 2 RESULTS FOR {data_set} \n" + 100 * "_")

            data = self.datasets[data_set]["x"]
            labels = self.datasets[data_set]["y"]

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)

            self.run_pca(scaled_data, data_set)
            self.run_ica(scaled_data, data_set)
            self.run_rca(scaled_data, data_set)
            self.run_lda(scaled_data, labels, data_set)

        self.kurtosis_plot('ICA', '2')
        self.kurtosis_plot('RCA', '2')

    def part1_experiment(self):
        for data_set in self.datasets:
            self.write_to_output(f"PART 1 RESULTS FOR {data_set} \n" + 100 * "_")

            data = self.datasets[data_set]["x"]
            labels = self.datasets[data_set]["y"]

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)

            (n_samples, n_features), n_labels = data.shape, np.unique(labels).size
            self.write_to_output(f"# labels: {n_labels}; # samples: {n_samples}; # features {n_features}")
            self.write_to_output(100 * "_")
            self.write_to_output("name\t\tclusters\ttime\tsilhouette")

            em_results = pd.DataFrame(columns=['clusters', 'fit_time', 'silhouette'])
            kmeans_results = pd.DataFrame(columns=['clusters', 'fit_time', 'silhouette'])

            # Silhouette Visualization code was taken from this resource with some modification
            # https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam
            fig, ax = plt.subplots(2, 2, figsize=(15, 8))
            plt.title(f'K-Means Silhouette Analysis for {data_set}')
            for n_clusters in self.clusters:
                q, mod = divmod(n_clusters, 2)
                kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10, random_state=self.random_seed)
                visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
                visualizer.fit(scaled_data)

            plt.savefig(f'images/{data_set}_KM_Silhouette_Analysis.png')
            plt.close()

            for n_clusters in self.clusters:
                kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10, random_state=self.random_seed)
                results = self.bench_cluster(algo=kmeans, name="k-means++", n_clusters=n_clusters, data=scaled_data)
                kmeans_results = pd.concat([kmeans_results, results])

                em = GaussianMixture(n_components=n_clusters, random_state=self.random_seed)
                results = self.bench_cluster(algo=em, name="em", n_clusters=n_clusters, data=scaled_data)
                em_results = pd.concat([em_results, results])

            self.plot_cluster_results(data_set, 'K-Means', kmeans_results)
            self.plot_cluster_results(data_set, 'Expectation Maximization', em_results)
            plt.savefig(f'images/{data_set}_Silhouette_Scores.png')
            plt.close()

            best_kmeans_n = kmeans_results.query('silhouette == silhouette.max()').clusters[0]
            best_em_n = em_results.query('silhouette == silhouette.max()').clusters[0]

            if data_set == 'Heart Data':
                x_vars = ['AgeCategory', 'Sex', 'Smoking', 'BMI', 'PhysicalActivity']
                y_vars = x_vars
            else:
                x_vars = ['Rainfall', 'Sunshine', 'RainToday', 'Temp3pm', 'Humidity3pm']
                y_vars = x_vars

            kmeans = KMeans(init="k-means++", n_clusters=best_kmeans_n, n_init=10, random_state=self.random_seed)
            kmeans.fit(scaled_data)
            kmeans_pred = kmeans.predict(scaled_data)
            data_kmeans = data.copy()
            data_kmeans['cluster'] = kmeans_pred
            sns.pairplot(data_kmeans.sample(1000), hue='cluster', palette='bright', x_vars=x_vars, y_vars=y_vars)
            plt.savefig(f"images/kmeans_pairplot_{data_set}.png", dpi=300)
            plt.close()

            kmeans_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=data.columns)
            self.write_to_output(100 * "_")
            self.write_to_output("K-Means Centers")
            self.write_to_output(kmeans_centers.to_string(header=True, index=False))
            kmeans_cm = pd.DataFrame(metrics.cluster.contingency_matrix(labels, kmeans_pred))

            self.write_to_output(100 * "_")
            self.write_to_output("K-Means Contingency Matrix")
            self.write_to_output(kmeans_cm.to_string(header=True, index=False))
            self.write_to_output(100 * "_")

            em = GaussianMixture(n_components=best_em_n, random_state=self.random_seed)
            em.fit(scaled_data)
            em_pred = em.predict(scaled_data)
            data_em = data.copy()
            data_em['cluster'] = em_pred
            sns.pairplot(data_em.sample(1000), hue='cluster', palette='bright', x_vars=x_vars, y_vars=y_vars)
            plt.savefig(f"images/em_pairplot_{data_set}.png", dpi=300)
            plt.close()

            em_means = pd.DataFrame(scaler.inverse_transform(em.means_), columns=data.columns)
            self.write_to_output(100 * "_")
            self.write_to_output("EM Means")
            self.write_to_output(em_means.to_string(header=True, index=False))

            em_cm = pd.DataFrame(metrics.cluster.contingency_matrix(labels, em_pred))
            self.write_to_output(100 * "_")
            self.write_to_output("EM Contingency Matrix")
            self.write_to_output(em_cm.to_string(header=True, index=False))
            self.write_to_output(100 * "_")


exp = experiments()
np.random.seed(exp.random_seed)
exp.prep_weather_data()
exp.prep_heart_data()
exp.part2_experiment()
# exp.part1_experiment()
# exp.neural_net_experiment()