from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    def generate_learning_curves(self, x_train, y_train, model, title, part):
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
        plt.savefig(f"images/part{part}/{file_name}.png")
        plt.close()

    def plot_validation_curve(self, train_scores, cv_scores, title, param_name, param_range, xscale, part):
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
        plt.savefig(f"images/part{part}/{file_name}.png")
        plt.close()

    def generate_validation_curves(self, x_train, y_train, model, hyperparams, title, part):
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
            self.plot_validation_curve(train_scores, cv_scores, title, param, param_range, xscale, part)

    def generate_loss_curve(self, x_train, y_train, model, title, part):
        model.fit(x_train, y_train)
        loss_curve = model.loss_curve_
        plt.plot(loss_curve, "o-", color="g", label="loss curve")
        plt.title(title)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.grid()
        file_name = title.replace(" ", "")
        plt.savefig(f"images/part{part}/{file_name}.png")
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

    def neural_net_experiment(self, data_set, x, y, part, algo_name):
        self.write_to_output(f"NEURAL NETWORK FOR {data_set} WITH {algo_name} REDUCED DATA \n ___________________________________________________")
        vc_title = f"Neural Network Validation Curve for {data_set} - {algo_name}"
        lc_title = f"Neural Network Learning Curve for {data_set} - {algo_name}"
        loss_title = f"Neural Network Loss Curve for {data_set} - {algo_name}"

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

        self.generate_validation_curves(x_train, y_train, self.nn_model, self.nn_hyperparameters, vc_title, part)
        self.tuned_nn = self.tune_hyperparams(x_train, y_train, self.nn_model, self.nn_hyperparameters, "Neural Network", data_set)
        self.generate_learning_curves(x_train, y_train, self.tuned_nn, lc_title, part)
        self.generate_loss_curve(x_train, y_train, self.tuned_nn, loss_title, part)
        self.get_test_performance(x_train, y_train, x_test, y_test, self.tuned_nn, part)

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
        self.write_to_output('Silhouette analysis')
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

    def plot_ev(self, model, algo_name, data_set, part):
        exp_var = model.explained_variance_ratio_
        eigen = np.cumsum(exp_var)
        plt.figure(algo_name)
        plt.bar(range(0, len(exp_var)), exp_var, alpha=0.5, align='center',
                label='Individual explained variance')
        plt.step(range(0, len(eigen)), eigen, where='mid',
                 label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.title(f"{algo_name} on {data_set}")
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'images/part{part}/{data_set}_{algo_name}_EV.png')
        plt.close()

    def run_pca(self, x, y, data_set, part):
        pca = PCA(random_state=self.random_seed)
        x_pca = pca.fit_transform(x)
        self.plot_ev(pca, 'PCA', data_set, part)

        x_train, x_test, y_train, y_test = self.split_data(x, y)
        pca.fit(x_train, y_train)
        self.write_to_output(f"RCA performance on {data_set} \n" + 100 * "_")
        self.score_dim_red(pca, x_test, y_test)
        self.bench_on_knn(pca, x_train, y_train, x_test, y_test)
        return x_pca

    def run_ica(self, x, y, data_set, part):
        plt.figure(f"ICA{part}")
        n_components = range(1, x.shape[1] + 1)
        kurtosis = pd.DataFrame(index=n_components, columns=['kurtosis'])
        for n in n_components:
            ica = FastICA(n_components=n, random_state=self.random_seed)
            k = pd.DataFrame(ica.fit_transform(x)).kurtosis().abs().mean()
            kurtosis.loc[n, 'kurtosis'] = k
        plt.plot(kurtosis, label=f"{data_set} kurtosis")

        n = kurtosis['kurtosis'].astype('float').idxmax()
        x_train, x_test, y_train, y_test = self.split_data(x, y)
        ica = FastICA(n_components=n, random_state=self.random_seed)
        ica.fit(x_train, y_train)
        self.write_to_output(f"ICA performance on {data_set} \n" + 100 * "_")
        self.bench_on_knn(ica, x_train, y_train, x_test, y_test)
        return ica.fit_transform(x)

    def run_rca(self, x, y, data_set, part):
        plt.figure(f"RCA{part}")
        n_components = range(1, x.shape[1] + 1)
        kurtosis = pd.DataFrame(index=n_components, columns=['kurtosis'])
        for n in n_components:
            rca = SparseRandomProjection(n_components=n, random_state=self.random_seed)
            k = pd.DataFrame(rca.fit_transform(x)).kurtosis().abs().mean()
            kurtosis.loc[n, 'kurtosis'] = k
        plt.plot(kurtosis, label=f"{data_set} Kurtosis")

        n = kurtosis['kurtosis'].astype('float').idxmax()
        x_train, x_test, y_train, y_test = self.split_data(x, y)
        rca = SparseRandomProjection(n_components=n, random_state=self.random_seed)
        rca.fit(x_train, y_train)
        self.write_to_output(f"RCA performance on {data_set} \n" + 100 * "_")
        self.bench_on_knn(rca, x_train, y_train, x_test, y_test)
        return rca.fit_transform(x)

    def run_lda(self, x, y, data_set, part):
        x_train, x_test, y_train, y_test = self.split_data(x, y)
        lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
        lda.fit(x_train, y_train)
        self.write_to_output(f"LDA performance on {data_set} \n" + 100 * "_")
        self.score_dim_red(lda, x_test, y_test)
        self.bench_on_knn(lda, x_train, y_train, x_test, y_test)
        return lda.fit_transform(x, y)

    def split_data(self, x, y):
        return train_test_split(
            x, y, test_size=0.2, random_state=self.random_seed
        )

    def score_dim_red(self, model, x_test, y_test):
        acc = model.score(x_test, y_test)
        self.write_to_output(f"Accuracy {acc}")

    def bench_on_knn(self, model, x_train, y_train, x_test, y_test):
        knn = KNeighborsClassifier(n_neighbors=50)
        knn.fit(model.transform(x_train), y_train)
        acc_knn = knn.score(model.transform(x_test), y_test)
        self.write_to_output(f"knn accuracy {acc_knn}")

    def kurtosis_plot(self, algorithm, part):
        plt.figure(algorithm+part)
        plt.ylabel('Kurtosis')
        plt.xlabel('n_components')
        plt.title(f"Average kurtosis for n_components for {algorithm}")
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'images/part{part}/{algorithm}_kurtosis.png')
        plt.close()

    def run_kmeans_sa(self, scaled_data, data_set, part, algo=''):
        # Silhouette Visualization code was taken from this resource with some modification
        # https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam
        plt.figure(part+algo+data_set)
        fig, ax = plt.subplots(2, 2, figsize=(15, 8))
        for n_clusters in self.clusters:
            q, mod = divmod(n_clusters, 2)
            kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10, random_state=self.random_seed)
            visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q - 1][mod])
            visualizer.fit(scaled_data)

        plt.savefig(f'images/part{part}/{data_set}_KM_Silhouette_Analysis_{algo}.png')
        plt.close()

    def run_kmeans_em(self, scaled_data, data_set, part, algo=''):
        plt.figure(part + algo + data_set)
        em_results = pd.DataFrame(columns=['clusters', 'fit_time', 'silhouette'])
        kmeans_results = pd.DataFrame(columns=['clusters', 'fit_time', 'silhouette'])

        for n_clusters in self.clusters:
            kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10, random_state=self.random_seed)
            results = self.bench_cluster(algo=kmeans, name="k-means++", n_clusters=n_clusters, data=scaled_data)
            kmeans_results = pd.concat([kmeans_results, results])

            em = GaussianMixture(n_components=n_clusters, random_state=self.random_seed)
            results = self.bench_cluster(algo=em, name="em", n_clusters=n_clusters, data=scaled_data)
            em_results = pd.concat([em_results, results])

        self.plot_cluster_results(data_set, 'K-Means', kmeans_results)
        self.plot_cluster_results(data_set, 'Expectation Maximization', em_results)

        plt.savefig(f'images/part{part}/{data_set}_Silhouette_Scores_{algo}.png')
        plt.close()

        best_kmeans_n = kmeans_results.query('silhouette == silhouette.max()').clusters[0]
        best_em_n = em_results.query('silhouette == silhouette.max()').clusters[0]
        return best_kmeans_n, best_em_n

    def generate_pair_plot(self, data, data_set, part, algo, algo2=''):
        if data_set == 'Heart Data':
            x_vars = ['AgeCategory', 'Sex', 'Smoking', 'BMI', 'PhysicalActivity']
            y_vars = x_vars
        else:
            x_vars = ['Rainfall', 'Sunshine', 'RainToday', 'Temp3pm', 'Humidity3pm']
            y_vars = x_vars

        sns.pairplot(data.sample(1000), hue='cluster', palette='bright', x_vars=x_vars, y_vars=y_vars)
        plt.savefig(f"images/part{part}/pairplot_{data_set}_{algo}_{algo2}.png", dpi=300)
        plt.close()

    def run_kmeans(self, n, scaled_data, data, data_set, part, algo=''):
        kmeans = KMeans(init="k-means++", n_clusters=n, n_init=10, random_state=self.random_seed)
        kmeans.fit(scaled_data)
        kmeans_pred = kmeans.predict(scaled_data)
        data_kmeans = data.copy()
        data_kmeans['cluster'] = kmeans_pred
        self.generate_pair_plot(data_kmeans, data_set, part, 'kmeans', algo)
        return kmeans, kmeans_pred

    def run_em(self, n, scaled_data, data, data_set, part, algo=''):
        em = GaussianMixture(n_components=n, random_state=self.random_seed)
        em.fit(scaled_data)
        em_pred = em.predict(scaled_data)
        data_em = data.copy()
        data_em['cluster'] = em_pred
        self.generate_pair_plot(data_em, data_set, part, 'em', algo)
        return em, em_pred

    def bench_kmeans(self, scaler, kmeans, columns, labels, kmeans_pred):
        if columns == None:
            kmeans_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_))
        else:
            kmeans_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=columns)
        self.write_to_output(100 * "_")
        self.write_to_output("K-Means Centers")
        self.write_to_output(kmeans_centers.to_string(header=True, index=False))

        kmeans_cm = metrics.multilabel_confusion_matrix(labels, kmeans_pred)
        self.write_to_output(100 * "_")
        self.write_to_output("K-Means Confusion Matrix")
        self.write_to_output(np.array2string(kmeans_cm))
        self.write_to_output(100 * "_")

    def bench_em(self, scaler, em, columns, labels, em_pred):
        if columns == None:
            em_means = pd.DataFrame(scaler.inverse_transform(em.means_))
        else:
            em_means = pd.DataFrame(scaler.inverse_transform(em.means_), columns=columns)
        self.write_to_output(100 * "_")
        self.write_to_output("EM Means")
        self.write_to_output(em_means.to_string(header=True, index=False))

        em_cm = metrics.multilabel_confusion_matrix(labels, em_pred)
        self.write_to_output(100 * "_")
        self.write_to_output("EM Confusion Matrix")
        self.write_to_output(np.array2string(em_cm))
        self.write_to_output(100 * "_")

    def data_prep(self, data_set):
        data = self.datasets[data_set]["x"]
        labels = self.datasets[data_set]["y"]

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        (n_samples, n_features), n_labels = data.shape, np.unique(labels).size
        self.write_to_output(f"# labels: {n_labels}; # samples: {n_samples}; # features {n_features}")
        self.write_to_output(100 * "_")
        return data, scaled_data, labels, scaler


    def part1_experiment(self):
        for data_set in self.datasets:
            self.write_to_output(f"PART 1 RESULTS FOR {data_set} \n" + 100 * "_")
            data, scaled_data, labels, scaler = self.data_prep(data_set)

            self.run_kmeans_sa(scaled_data, data_set, '1')
            best_kmeans_n, best_em_n = self.run_kmeans_em(scaled_data, data_set, '1')

            kmeans, kmeans_pred = self.run_kmeans(best_kmeans_n, scaled_data, data, data_set, '1')
            em, em_pred = self.run_em(best_em_n, scaled_data, data, data_set, '1')

            self.bench_kmeans(scaler, kmeans, data.columns, labels, kmeans_pred)
            self.bench_em(scaler, em, data.columns, labels, em_pred)

    def part2_experiment(self):
        for data_set in self.datasets:
            self.write_to_output(f"\nPART 2 RESULTS FOR {data_set} \n" + 100 * "_")

            data = self.datasets[data_set]["x"]
            labels = self.datasets[data_set]["y"]

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)

            self.run_pca(scaled_data, labels, data_set, '2')
            self.run_ica(scaled_data, labels, data_set, '2')
            self.run_rca(scaled_data, labels, data_set, '2')
            self.run_lda(scaled_data, labels, data_set, '2')

        self.kurtosis_plot('ICA', '2')
        self.kurtosis_plot('RCA', '2')

    def part3_experiment(self):
        for data_set in self.datasets:
            self.write_to_output(f"\nPART 3 RESULTS FOR {data_set} \n" + 100 * "_")
            data, scaled_data, labels, scaler = self.data_prep(data_set)

            pca = self.run_pca(scaled_data, labels, data_set, '3')
            ica = self.run_ica(scaled_data, labels, data_set, '3')
            rca = self.run_rca(scaled_data, labels, data_set, '3')
            lda = self.run_lda(scaled_data, labels, data_set, '3')

            reduced_data = {'pca': pca, 'ica': ica, 'rca': rca, 'lda': lda}

            for i, (algo_name, reduced) in enumerate(reduced_data.items()):
                self.write_to_output(f"RESULTS FOR {algo_name}")
                self.run_kmeans_sa(reduced, data_set, '3', algo_name)
                best_kmeans_n, best_em_n = self.run_kmeans_em(reduced, data_set, '3', algo_name)

                kmeans, kmeans_pred = self.run_kmeans(best_kmeans_n, reduced, data, data_set, '3', algo_name)
                em, em_pred = self.run_em(best_em_n, reduced, data, data_set, '3', algo_name)

                self.bench_kmeans(scaler, kmeans, None, labels, kmeans_pred)
                self.bench_em(scaler, em, None, labels, em_pred)

        self.kurtosis_plot('ICA', '3')
        self.kurtosis_plot('RCA', '3')

    def part4_experiment(self):
        data_set = 'Weather Data'
        self.write_to_output(f"PART 4 RESULTS FOR {data_set} \n" + 100 * "_")
        data, scaled_data, labels, scaler = self.data_prep(data_set)

        pca = self.run_pca(scaled_data, labels, data_set, '4')
        ica = self.run_ica(scaled_data, labels, data_set, '4')
        rca = self.run_rca(scaled_data, labels, data_set, '4')
        lda = self.run_lda(scaled_data, labels, data_set, '4')

        self.kurtosis_plot('ICA', '4')
        self.kurtosis_plot('RCA', '4')

        reduced_data = {'pca': pca, 'ica': ica, 'rca': rca, 'lda': lda}

        for i, (algo_name, reduced) in enumerate(reduced_data.items()):
            self.write_to_output(f"{algo_name} \n" + 100 * "_")
            self.neural_net_experiment(data_set, reduced, labels, algo_name, '4')

        self.write_to_output(f"Original Data \n" + 100 * "_")
        self.neural_net_experiment(data_set, scaled_data, labels, 'Original Data', '4')

exp = experiments()
np.random.seed(exp.random_seed)
exp.prep_weather_data()
exp.prep_heart_data()
# exp.part1_experiment()
# exp.part2_experiment()
exp.part3_experiment()
exp.part4_experiment()