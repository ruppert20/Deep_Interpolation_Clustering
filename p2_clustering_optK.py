#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
clustering.py: 
"""
import os
import os.path as osp
import argparse
import numpy as np
from internal_eval import Sihouette, DBIndex, CHIndex, DunnIndex
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances, silhouette_score
from scipy.spatial.distance import cdist
from scipy.stats import chi2_contingency
import gc
import pandas as pd
from kneed import KneeLocator
from utils import print_dict_byline, logger
from info import BASE_PATH, LEGEND_INFO, COHORTS
np.random.seed(123)

##The Silhouette score is bounded from -1 to 1 and higher score means more distinct clusters.
##The Calinski-Harabasz index compares the variance between-clusters to the variance within each cluster. This measure is much simpler to calculate then the Silhouette score however it is not bounded. The higher the score the better the separation is.
##Davies-Bouldin index is the ratio between the within cluster distances and the between cluster distances and computing the average overall the clusters. It is therefore relatively simple to compute, bounded – 0 to 1, lower score is better. 

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_method', default='kmeans', choices=['kmeans', 'dbscan', 'dl', 'optics', 'consensus'],
                        help='For consensus, the labels are generated outside, and then directly loaded.')
    parser.add_argument('--k_max', type=int, default=10, help='The max value of k, for k-means only.')
    parser.add_argument('--select_opt_k', default=['gap_sts', 'elbow', 'outcome'])
    parser.add_argument('--select_eps', type=str, default='k_distance_graph', help='Select eps for DBSCAN')
    parser.add_argument('--n_init', type=int, default=10, help='The number of initialization for k-means.')
    parser.add_argument('--gap_b', type=int, default=10, help='The number of randomly sampling for gap-sts.')
    parser.add_argument('--restore_metric', default=['ae_mse', 'loss'])
    parser.add_argument('--opt_eps', type=float, default=1.9, help="The optimal eps value for DBSCAN.")
    parser.add_argument('--internal_metrics', default=["Sihouette", "Davies-Bouldin_Index", "Calinski-Harabasz"])
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing plot files')
    args = parser.parse_args()
    return args


class Cluster(object):
    def __init__(self, args):
        self.args = args
        self.exp_path = os.path.join(os.getcwd(), "Results", "Pretrain")
    
    ###load hidden feature (numpy version)
    def load_data(self):
        cohort_lst = []
        for cohort in COHORTS:
            logger.info("*******Load the features for {}*******".format(cohort))
            full_data = np.load(osp.join(self.feat_path, "{}.npy".format(cohort)), allow_pickle=True).item()
            cohort_data = {kept_k: full_data[kept_k] for kept_k in ['encounter_id', 'hidden', 'ob', 'padding_mask']}
            cohort_lst.append(cohort_data)
            logger.debug(cohort_data.keys())
            encounter_id = cohort_data['encounter_id']
            logger.info('Cohort: {}, Sample: {}'.format(cohort, len(encounter_id)))
        self.train_data, self.valid_data, self.test_data = cohort_lst
        self.feat_dim = self.train_data['hidden'].shape[-1]

    ###This is used to select the optimal number of clusters
    def select_opt_k(self):
        """
        This is used for selecting the optimal number of clusters.
        :return:
        """
        for metric in self.args.restore_metric:
            self.feat_path = osp.join(self.exp_path, "out_feat", metric)
            self.out_path = osp.join(self.exp_path, 'out_feat', '{}_{}'.format(metric, self.args.cluster_method))
            self.out_path = self.out_path + '_aligned'
            os.makedirs(self.out_path, exist_ok=True)
            self.load_data()
                    
            if self.args.cluster_method == 'kmeans':
                print('kmeans')
                k_max = self.args.k_max
                km = KM(k_max, self.out_path, self.args.internal_metrics, self.args.n_init, self.args.gap_b)
                km.train(self.train_data, self.valid_data, self.args.select_opt_k, overwrite=self.args.overwrite)
            elif self.args.cluster_method == 'dbscan':
                eps_range = np.arange(.5, 5.1, .5)  # v11_2: 1.8-2.0; v8_1: 2.0-3.0
                db = Dbscan(eps_range=eps_range, min_samples=self.feat_dim+1, out_path=self.out_path)
                db.train(self.train_data, self.valid_data, self.args.select_eps, overwrite=self.args.overwrite)
            elif self.args.cluster_method == 'optics':
                op = Optics(min_samples=self.feat_dim+1, cluster_method='xi', out_path=self.out_path)   # cluster_method='dbscan'
                op.train(self.train_data, self.valid_data, overwrite=self.args.overwrite)

class Dbscan(object):
    def __init__(self, eps_range, min_samples, out_path):
        self.eps_range = eps_range
        self.min_sample = min_samples
        self.out_path = osp.join(out_path, 'plot')
        os.makedirs(self.out_path, exist_ok=True)

    def train(self, train_data, valid_data, select_eps, **kwargs):
        overwrite = kwargs.get('overwrite', False)
        train_feat, _ = train_data['hidden'], valid_data['hidden']
        train_feat_dist = pairwise_distances(train_feat)

        if select_eps == 'k_distance_graph':
            k = self.min_sample - 1
            plot_png = osp.join(self.out_path, "{}-NN distance.png".format(k))
            if osp.exists(plot_png) and not overwrite:
                logger.info(
                    "Not saved for {}! Because files existed and not allowed for overwrite.".format(plot_png))

            else:
                nbrs = NearestNeighbors(n_neighbors=k, n_jobs=5).fit(train_feat)
                distances, indices = nbrs.kneighbors(train_feat)
                k_distances = distances[:, -1]
                sorted_dist = sorted(k_distances)
                point_num = np.arange(1, len(sorted_dist)+1)
                df = pd.DataFrame.from_dict({'dist': sorted_dist, 'sample': point_num})

                # Calculate the elbow, refer to https://github.com/arvkevi/kneed
                kneedle = KneeLocator(point_num, sorted_dist, S=1.0, curve='convex', direction='increasing')
                elbow_x, elbow_y = kneedle.elbow, kneedle.elbow_y
                logger.info('The detected elbow: x: {}, y: {}'.format(elbow_x, elbow_y))

                sns.set(style="whitegrid")
                sns.set_context("poster")
                fig = plt.figure(figsize=(18, 12))
                ax = fig.add_subplot(1, 1, 1)

                sl = sns.lineplot(x='sample', y='dist', data=df, palette="tab10", linewidth=3, legend=False, ax=ax)  # **{'markersize': 30}
                # handles, labels = ax.get_legend_handles_labels()
                # leg = ax.legend(handles=handles, labels=labels, loc='best', ncol=1, borderaxespad=0., fontsize=30)

                sl.set_xlabel('Samples sorted by distance', fontsize=40)
                sl.set_ylabel('{}-NN distance'.format(k), fontsize=40)
                sl.tick_params(axis='both', labelsize=35)

                # fig.show()

                fig.savefig(plot_png, bbox_inches='tight')  # , dpi=300
                logger.info("Saved for {}!.".format(plot_png))

        for eps in self.eps_range:
            logger.info('\nRunning eps: {}'.format(eps))
            # db = DBSCAN(eps, self.min_sample).fit(train_feat)
            db = DBSCAN(eps, self.min_sample, metric='precomputed').fit(train_feat_dist)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            logger.info('core_sample: {}'.format(sum(core_samples_mask)))

            # Number of clusters in labels, ignoring noise if present.
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            # Denoise samples
            denoise_idx = labels != -1
            denoise_train_feat = train_feat[denoise_idx]
            denoise_label = labels[denoise_idx]

            logger.info('Estimated number of clusters: %d' % n_clusters_)
            logger.info('Estimated number of noise points: %d' % n_noise_)
            if n_clusters_ == 0:
                continue
            elif n_clusters_ == 1:
                logger.info('Skip the Silhouette Coefficient calculation.')
            else:
                logger.info('Orginal Sample: {} Silhouette Coefficient: {:.5f}'.
                            format(len(labels), silhouette_score(train_feat, labels)))
                logger.info('Denoise sample: {}, Denoise Silhouette Coefficient: {:.5f}'.
                            format(len(denoise_label), silhouette_score(denoise_train_feat, denoise_label)))


class Optics(object):
    def __init__(self, min_samples, cluster_method, out_path):
        self.min_sample = min_samples
        self.cluster_method = cluster_method
        self.out_path = osp.join(out_path, 'plot')
        os.makedirs(self.out_path, exist_ok=True)

    def train(self, train_data, valid_data, **kwargs):
        overwrite = kwargs.get('overwrite', False)
        train_feat, _ = train_data['hidden'], valid_data['hidden']
        train_feat_dist = pairwise_distances(train_feat)
        train_num = len(train_feat)

        plot_png = osp.join(self.out_path, "Reachability_{}.png".format(self.cluster_method))
        if osp.exists(plot_png) and not overwrite:
            logger.info(
                "Not saved for {}! Because files existed and not allowed for overwrite.".format(plot_png))
        else:
            if self.cluster_method == 'xi':
                optics = OPTICS(min_samples=self.min_sample, cluster_method=self.cluster_method, xi=.05,
                                min_cluster_size=self.min_sample, n_jobs=30)
            elif self.cluster_method == 'dbscan':
                optics = OPTICS(min_samples=self.min_sample, cluster_method=self.cluster_method, n_jobs=30)

            # optics.fit(train_feat)
            optics.fit(train_feat_dist)

            # Number of clusters in labels, ignoring noise if present.
            labels = optics.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            logger.info('OPTICS with cluster_method: {}, n_clusters: {}, n_noise: {}'.format(self.cluster_method, n_clusters_, n_noise_))

            # Plot reachability
            space = np.arange(train_num)
            reachability = optics.reachability_[optics.ordering_]
            labels = optics.labels_[optics.ordering_]

            fig = plt.figure(figsize=(18, 12))
            ax = fig.add_subplot(1, 1, 1)

            df = pd.DataFrame.from_dict({'x': space, 'dist': reachability, 'label': labels})
            df = df[df['label'] != -1]
            sc = sns.scatterplot(x='x', y='dist', hue='label', data=df, ax=ax)

            sc.set_xlabel('Reachability (epsilon distance)', fontsize=40)
            sc.set_ylabel('Samples', fontsize=40)
            sc.tick_params(axis='both', labelsize=35)

            fig.savefig(plot_png, bbox_inches='tight')  # , dpi=300
            logger.info("Saved for {}!.".format(plot_png))
            plt.close(fig)


class KM(object):
    def __init__(self, k_max, out_path, internal_metrics, n_init, gap_b):
        self.k_max = k_max
        self.out_path = osp.join(out_path, 'plot')
        os.makedirs(self.out_path, exist_ok=True)
        self.internal_metrics_names = internal_metrics
        self.internal_metrics = self.get_internal_metrics()
        self.n_init = n_init
        self.gap_b = gap_b

    def get_internal_metrics(self):
        metrics = []
        for metric_name in self.internal_metrics_names:
            if metric_name == "Dunn_Index":
                metric = DunnIndex()
            elif metric_name == "Sihouette":
                metric = Sihouette()
            elif metric_name == "Davies-Bouldin_Index":
                metric = DBIndex()
            elif metric_name == "Calinski-Harabasz":
                metric = CHIndex()
            metrics.append(metric)
        return metrics

    def train(self, train_data, valid_data, select_opt_k, **kwargs):
        overwrite = kwargs.get('overwrite', False)
        train_feat, valid_feat = train_data['hidden'], valid_data['hidden']

        for method in select_opt_k:
            if method == 'outcome':
                self.outcome_discrimination(train_data, overwrite=overwrite)
            elif method == 'elbow':
                rng = range(2, self.k_max + 1)
                train_distortions, valid_distortions = [], []
                for k in rng:
                    logger.info('Running K: {}'.format(k))
                    kmeans_model = KMeans(n_clusters=k, init='k-means++').fit(train_feat)
                    train_distortions.append(sum(np.min(cdist(train_feat, kmeans_model.cluster_centers_, 'euclidean'), axis=1))
                                             / train_feat.shape[0])
                    valid_distortions.append(sum(np.min(cdist(valid_feat, kmeans_model.cluster_centers_, 'euclidean'), axis=1))
                                             / valid_feat.shape[0])
                    gc.collect()
    
                for cohort, distortions in zip(['train', 'valid'], [train_distortions, valid_distortions]):
                    plt.figure()
                    plt.plot(list(rng), distortions, 'bx-')
                    plt.xlabel('Cluster Count', fontsize=18)
                    plt.ylabel('Distortion', fontsize=18)
                    plt.title('The Elbow method showing the optimal k', fontsize=20)
                    plt.savefig(os.path.join(self.out_path, '{}_elbow.png'.format(cohort)))
                    plt.close()
    
            elif method == 'gap_sts':
                for gap_sts_version in [1]:
                    # Gap-statistic uses train_feat only, not use valid_feat
                    sts_csv = osp.join(self.out_path, 'gap_sts_v{}.csv'.format(gap_sts_version))
                    if osp.exists(sts_csv) and not overwrite:
                        logger.info('Load the previous gat_sts.csv')
                        sts_df = pd.read_csv(sts_csv)
                    else:
                        sts_df = self.compute_gap_internal_metric(KMeans(n_init=self.n_init), train_feat, self.k_max,
                                                                  n_references=self.gap_b, version=gap_sts_version)
                        sts_df = sts_df.astype(float)
                        sts_df.to_csv(osp.join(self.out_path, 'gap_sts_v{}.csv'.format(gap_sts_version)), index=False)
        
                    melt_df = pd.melt(sts_df, id_vars='k', var_name='metrics',
                                      value_vars=['gap', 'ref', 'act'], value_name='value')
        
                    plot_f_name_1 = 'gap_statistic-1_v{}'.format(gap_sts_version)   # Only gap
                    plot_f_name_2 = 'gap_statistic-2_v{}'.format(gap_sts_version)   # Contain gap, act, ref
        
                    for i, plot_f_name in enumerate([plot_f_name_1, plot_f_name_2]):
                        plot_png = osp.join(self.out_path, "{}.png".format(plot_f_name))
                        # plot_eps = osp.join(self.out_path, "{}.eps".format(plot_f_name))
        
                        sns.set(style="whitegrid")
                        sns.set_context("poster")
                        fig = plt.figure(figsize=(18, 12))
                        ax = fig.add_subplot(1, 1, 1)
        
                        if i == 0:
                            sl = sns.lineplot(x='k', y='gap', data=sts_df, palette="tab10",
                                              marker='o', dashes=False, linewidth=3, ax=ax)  # **{'markersize': 30}
                            handles, labels = ax.get_legend_handles_labels()
                            leg = ax.legend(handles=handles, labels=labels, loc='best', ncol=1, borderaxespad=0., fontsize=30)
                        elif i == 1:
                            sl = sns.lineplot(x='k', y='value', hue='metrics', style='metrics', data=melt_df, palette="tab10",
                                              markers=True, dashes=False, linewidth=3, ax=ax)     #  **{'markersize': 30}
        
                            handles, labels = ax.get_legend_handles_labels()
                            leg = ax.legend(handles=handles[1:], labels=labels[1:], loc=2, ncol=1, borderaxespad=0., fontsize=30,
                                            bbox_to_anchor=(1.05, 1))
                        for t in leg.texts:
                            inside_df_legend_label = t.get_text()
                            t.set_text(LEGEND_INFO.get(inside_df_legend_label, inside_df_legend_label))
                            logger.debug(t.get_text())
        
                        sl.set_xlabel('Number of clusters K', fontsize=40)
                        sl.set_ylabel(LEGEND_INFO.get('log(inertia)', 'log(inertia)'), fontsize=40)
                        sl.tick_params(axis='both', labelsize=35)
                        plt.xticks(list(range(0, self.k_max + 1, 2)))
        
                        # fig.show()
        
                        if osp.exists(plot_png) and not overwrite:
                            logger.info("Not saved for {}! Because files existed and not allowed for overwrite.".format(plot_f_name))
                        else:
                            fig.savefig(plot_png, bbox_extra_artists=(leg,), bbox_inches='tight')  # , dpi=300
                            logger.info("Saved for {}!.".format(plot_f_name))

    def compute_inertia_v1(self, a, X):
        """
        This implementation follows
        :param a:
        :param X:
        :return:
        """
        W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
        return np.mean(W)

    def computer_intertia_v2(self, a, X):
        wk = 0
        for c in np.unique(a):
            dr = np.sum(pairwise_distances(X[a == c, :]))   # Sum 2 times pairwise dist except 0 on the diagonal.
            # dr = np.sum(np.triu(pairwise_distances(X[a == c, :])))  # Sum 1 time pairwise dist, but it does not matter
            nr = (a == c).sum()
            wk = wk + dr/(2*nr)
        return wk

    def compute_gap_internal_metric(self, clustering, data, k_max=5, n_references=5, version=2):
        """
        https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        data_rng = data.max() - data.min()
        logger.info('Data max: {}, min: {}, rng: {}'.format(data.max(), data.min(), data_rng))
        k_rng = range(2, k_max + 1)
        vals = pd.DataFrame(index=k_rng, columns=['k', 'gap', 'ref', 'act', 'ref_s'] + self.internal_metrics_names)

        for k in k_rng:
            local_inertia = []
            clustering.n_clusters = k
            if version == 1:
                for _ in range(n_references):
                    reference = np.random.random_sample(data.shape) * data_rng + data.min()
                    # reference = np.random.rand(*data.shape)
                    assignments = clustering.fit_predict(reference)
                    local_inertia.append(self.compute_inertia_v1(assignments, reference))
                ref, ref_std = np.mean(np.log(local_inertia)), np.std(np.log(local_inertia))
                ref_s = np.sqrt(1 + 1/n_references) * ref_std

                assignments = clustering.fit_predict(data)
                act = self.compute_inertia_v1(assignments, data)
                act = np.log(act)

            elif version == 2:
                for _ in range(n_references):
                    reference = np.random.random_sample(data.shape) * data_rng + data.min()
                    assignments = clustering.fit_predict(reference)
                    local_inertia.append(self.computer_intertia_v2(assignments, reference))
                ref, ref_std = np.mean(np.log(local_inertia)), np.std(np.log(local_inertia))
                ref_s = np.sqrt(1 + 1 / n_references) * ref_std

                assignments = clustering.fit_predict(data)
                act = self.computer_intertia_v2(assignments, data)
                act = np.log(act)

            gap = ref - act
            gap_sts_str = 'k: {}, gap: {:.4f}, ref: {:.4f}, act: {:.4f}, ref_s: {:.4f}'.format(k, gap, ref, act, ref_s)

            # compute the internal metric
            if k == 1:
                metric_str = ' When k == 1, no internal metric is available.'
                metric_values = [0] * len(self.internal_metrics_names)
            else:
                metric_str, metric_values = '', []
                for internal_metric_name, internal_metric in zip(self.internal_metrics_names, self.internal_metrics):
                    metric_value = internal_metric(data, assignments)
                    metric_str += ' {}: {:.4f}'.format(internal_metric_name, metric_value)
                    metric_values.append(metric_value)

            vals.loc[k] = [k, gap, ref, act, ref_s] + metric_values
            info_str = gap_sts_str + metric_str
            logger.info(info_str)
        return vals

    def outcome_discrimination(self, train_data, overwrite=False):
        """Evaluate each K by how well clusters separate the combined endpoint.

        For each K, runs k-means, builds a cluster x endpoint contingency table,
        and computes chi-squared statistic, p-value, risk ratio (max/min cluster
        event rate), and absolute risk spread (max - min).  Results are saved to
        CSV and plotted.
        """
        csv_path = osp.join(self.out_path, 'outcome_discrimination.csv')
        if osp.exists(csv_path) and not overwrite:
            logger.info('Loading existing outcome discrimination results from {}'.format(csv_path))
            return pd.read_csv(csv_path)

        # Load outcome labels matched to training encounter IDs
        table_data_path = osp.join(BASE_PATH, 'Data', 'analysis_data', 'table_data.csv')
        outcomes = pd.read_csv(table_data_path).rename(columns={'encounter_deiden_id': 'encounter_id'})
        outcomes['encounter_id'] = outcomes['encounter_id'].astype(str)
        outcomes['combined_endpoint_bin'] = (outcomes['combined_endpoint'] == 'Y').astype(int)
        outcomes['ICU_bin'] = (outcomes['ICU'] == 'Y').astype(int)
        outcomes['rapid_response_bin'] = (outcomes['rapid_response'] == 'Y').astype(int)

        train_feat = train_data['hidden']
        train_ids = np.array(train_data['encounter_id']).astype(str)

        k_rng = range(2, self.k_max + 1)
        rows = []

        for k in k_rng:
            km = KMeans(n_clusters=k, n_init=self.n_init, init='k-means++').fit(train_feat)
            labels = km.labels_

            df = pd.DataFrame({'encounter_id': train_ids, 'cluster': labels})
            df = df.merge(outcomes, on='encounter_id', how='inner')

            results = {'k': k}

            for endpoint, col in [('combined_endpoint', 'combined_endpoint_bin'),
                                  ('ICU', 'ICU_bin'),
                                  ('rapid_response', 'rapid_response_bin')]:
                ct = pd.crosstab(df['cluster'], df[col])

                # Ensure both columns exist (0 and 1)
                if ct.shape[1] < 2:
                    results['{}_chi2'.format(endpoint)] = 0.0
                    results['{}_pvalue'.format(endpoint)] = 1.0
                    results['{}_risk_ratio'.format(endpoint)] = 1.0
                    results['{}_risk_spread'.format(endpoint)] = 0.0
                    continue

                chi2, p_value, _, _ = chi2_contingency(ct)

                cluster_rates = ct[1] / ct.sum(axis=1)
                min_rate = cluster_rates.min()
                max_rate = cluster_rates.max()
                risk_ratio = max_rate / min_rate if min_rate > 0 else float('inf')
                risk_spread = max_rate - min_rate

                results['{}_chi2'.format(endpoint)] = chi2
                results['{}_pvalue'.format(endpoint)] = p_value
                results['{}_risk_ratio'.format(endpoint)] = risk_ratio
                results['{}_risk_spread'.format(endpoint)] = risk_spread

                logger.info('K={}: {} chi2={:.2f}, p={:.4f}, risk_ratio={:.2f}, '
                            'spread={:.4f} (min={:.4f}, max={:.4f})'.format(
                    k, endpoint, chi2, p_value, risk_ratio, risk_spread, min_rate, max_rate))

            rows.append(results)

        result_df = pd.DataFrame(rows)
        result_df.to_csv(csv_path, index=False)
        logger.info('Outcome discrimination results saved to {}'.format(csv_path))

        # Plot: risk ratio and chi2 for combined_endpoint across K
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('Outcome-Aware K Selection', fontsize=22)

        ks = result_df['k'].values

        ax = axes[0, 0]
        ax.plot(ks, result_df['combined_endpoint_chi2'], 'o-', linewidth=2, color='#e74c3c')
        ax.set_xlabel('K', fontsize=16)
        ax.set_ylabel('Chi-squared', fontsize=16)
        ax.set_title('Combined Endpoint: Chi-squared', fontsize=18)
        ax.set_xticks(ks)

        ax = axes[0, 1]
        ax.plot(ks, result_df['combined_endpoint_risk_ratio'], 's-', linewidth=2, color='#3498db')
        ax.set_xlabel('K', fontsize=16)
        ax.set_ylabel('Risk Ratio (max/min)', fontsize=16)
        ax.set_title('Combined Endpoint: Risk Ratio', fontsize=18)
        ax.set_xticks(ks)

        ax = axes[1, 0]
        for endpoint, color in [('combined_endpoint', '#e74c3c'), ('ICU', '#3498db'), ('rapid_response', '#2ecc71')]:
            ax.plot(ks, result_df['{}_risk_spread'.format(endpoint)], 'o-', linewidth=2, color=color, label=endpoint)
        ax.set_xlabel('K', fontsize=16)
        ax.set_ylabel('Risk Spread (max - min rate)', fontsize=16)
        ax.set_title('Risk Spread Across Endpoints', fontsize=18)
        ax.set_xticks(ks)
        ax.legend(fontsize=14)

        ax = axes[1, 1]
        for endpoint, color in [('combined_endpoint', '#e74c3c'), ('ICU', '#3498db'), ('rapid_response', '#2ecc71')]:
            pvals = result_df['{}_pvalue'.format(endpoint)]
            ax.plot(ks, -np.log10(pvals.clip(lower=1e-20)), 'o-', linewidth=2, color=color, label=endpoint)
        ax.axhline(-np.log10(0.05), color='gray', linestyle='--', label='p=0.05')
        ax.set_xlabel('K', fontsize=16)
        ax.set_ylabel('-log10(p-value)', fontsize=16)
        ax.set_title('Statistical Significance', fontsize=18)
        ax.set_xticks(ks)
        ax.legend(fontsize=14)

        plt.tight_layout()
        plot_path = osp.join(self.out_path, 'outcome_discrimination.png')
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        logger.info('Outcome discrimination plot saved to {}'.format(plot_path))

        return result_df


def main(args):
    cluster = Cluster(args)
    cluster.select_opt_k()

if __name__ == '__main__':
    args = get_arguments()
    print_dict_byline(vars(args))
    main(args)
