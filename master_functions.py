import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA

def load_in_taxonomy_json():
    import json
    import pathlib

    orgs = {}
    for file in pathlib.Path('./taxonomy/').iterdir():
        with open(file, 'r') as open_file:
            orgs[file.stem] = json.load(open_file)
    return orgs


def get_correct_preds_percentages(df, orgs: dict, row_name: str):
    ranks = ['species', 'genus', 'family', 'order', 'class', 'phylum', 'superkingdom']
    # groupby viruses, within these groups find rows with max values in '1' and return the indecies
    mask = df.groupby('virus')['1'].transform(max) == df['1']
    # create a df for 'best' predictions
    df_best = df.loc[mask, ['virus', 'host', '1']]

    def lookup_taxonomy(x, rank: str, orgs: dict):
        rank_idx = orgs['host'][x['host']]['lineage_ranks'].index(rank)
        return 1 if orgs["host"][x['host']]["lineage_names"][rank_idx] == \
            orgs['virus'][x['virus']]['host']['lineage_names'][rank_idx] else 0

    for rank in ranks:
        df_best[f'{rank}_correct'] = df_best.apply(lookup_taxonomy, rank=rank, orgs=orgs, axis=1)
        df_best[f'{rank}_correct'] = df_best[f'{rank}_correct'].astype('bool')
    df_best.drop(['virus', 'host', '1'], axis=1, inplace=True)
    df_to_return = pd.DataFrame(index=[f'{row_name}'], columns=df_best.columns)
    for rank in ranks:
        df_to_return[f'{rank}_correct'] = df_best[f"{rank}_correct"].sum() / len(df_best.index)*100
    return df_to_return


from sklearn.preprocessing import Normalizer
def get_probabilities(features: list, results_selected, large_df: pd.DataFrame, normalize: bool = False):
    '''
    Get probabilities for each observation in each X subset in cross validation individually
    '''
    groups_main = large_df['group_code'].values
    # get the probabilities
    X_sel_main = large_df[features]
    prob_df_sel = pd.DataFrame(index=range(len(large_df['y'])), columns=['0', '1'])
    prob_df_sel['0'] = prob_df_sel['0'].astype('float')
    prob_df_sel['1'] = prob_df_sel['1'].astype('float')
    for i in range(0, max(groups_main)+1):
        mask_main = groups_main == i
        X_curr = X_sel_main.loc[mask_main,:]
        if normalize:
            X_curr = Normalizer().fit_transform(X_sel_main.loc[mask_main,:])
        prob_df_sel.loc[mask_main, ['0', '1']] = \
            results_selected['estimator'][i].predict_proba(X_curr)
    return prob_df_sel


from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.model_selection import LeaveOneGroupOut

# Classification and ROC analysis
def draw_roc_cv(df_to_draw: pd.DataFrame, cv_splitter, X_main: pd.DataFrame, y_main: pd.DataFrame, groups_main, est_name: str):
    # Run classifier with cross-validation and plot ROC curves
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(20, 20))
    for i, (train, test) in enumerate(cv_splitter.split(X_main, y_main, groups=groups_main)):
        viz = RocCurveDisplay.from_predictions(
            y_true=df_to_draw.loc[test, 'y'],
            y_pred=df_to_draw.loc[test, '1'],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"{est_name} - ROC curve",
    )
    ax.legend(loc="lower right")
    plt.show()
    return


def draw_rocs(dfs: list, names: list, plot_name: str):
    plt.figure(figsize=(12, 7))
    for i, df in enumerate(dfs):
        fpr, tpr, thresh = roc_curve(df['y'], df['1'])
        plt.plot(fpr, tpr, label=f'AUC ({names[i]}) = {auc(fpr, tpr):.2f}')

    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    plt.title(f'{plot_name}', size=20)
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend()


def draw_precision_recall_curve(dfs: list, names: list, plot_name: str):
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    for i, df in enumerate(dfs):
        color = next(ax._get_lines.prop_cycler)['color']
        precision, recall, thresholds = precision_recall_curve(df['y'], df['1'])
        auc_temp = auc(recall, precision)
        f1_scores = np.divide(2*recall*precision, recall+precision)
        f1_scores = np.nan_to_num(f1_scores)
        # f1_scores = 2*recall*precision/(recall+precision)
        index = np.argmax(f1_scores)

        line = plt.plot(recall, precision, label=f'AUC ({names[i]}) = {auc_temp:.2f}', color=color)
        plt.plot(recall[index], precision[index], marker='o', color=color) # marker
        plt.text(recall[index], precision[index]+(0.03 if i%2 == 0 else -0.05), \
            # f'x={recall[index]:.2f}, y={precision[index]:.2f}', color=color)
            f'max_f1=({np.amax(f1_scores):.2f})', color=color)
    plt.title(f'{plot_name}', size=20)
    plt.xlabel('Recall', size=14)
    plt.ylabel('Precision', size=14)
    plt.legend()
    return


def draw_gini_importances(importances, std, X):
    forest_importances = pd.Series(importances, index=X.columns)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    return


def get_shap_values(estimators: list, X_learn: pd.DataFrame, learning_df: pd.DataFrame):
    import shap
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

    vals = []
    for i, est in enumerate(estimators):
        X_test = X_learn[learning_df['group_code'] == i].values
        explainer = shap.Explainer(est.predict, X_learn[learning_df['group_code'] != i].values)
        vals.append(explainer(X_test))
    vals = np.array(vals)
    return vals


def plot_shap_values(shap_values: list, X_learn: pd.DataFrame, learning_df: pd.DataFrame):
    import shap
    for i, vals in enumerate(shap_values):
        shap.summary_plot(vals, X_learn[learning_df['group_code'] == i], show=False)
        plt.title(f'Group {i} ({learning_df[learning_df["group_code"] == i].iloc[0]["group"]}) SHAP values')
        plt.savefig(f'test/shap_plots/group_{i}.png')
        plt.show()
        plt.clf()
    return


def plot_mean_shap(X_learn: pd.DataFrame, shap_values):
    plt.bar(X_learn.columns, np.mean([np.mean(np.absolute(i), axis=0) for i in shap_values], axis=0))
    plt.xticks(rotation=90)
    plt.ylabel('Mean absolute SHAP value')
    plt.title('Random Forest - mean absolute SHAP values')


def get_shap_values_lrc(estimators: list, X_learn: pd.DataFrame, learning_df: pd.DataFrame):
    import shap
    vals = []
    for i, est in enumerate(estimators):
        X_test = X_learn[learning_df['group_code'] == i]
        explainer = shap.Explainer(est, X_test)
        vals.append(explainer(X_test))
    vals = np.array(vals)
    return vals


def get_pca(X, y, components: int):
    # check for normal distribution and scale appropriately
    X_pca = Normalizer().fit_transform(X)
    # perform PCA
    pca = PCA(n_components=components)
    X_pca = pca.fit_transform(X_pca)
    cols = [f'component {i+1}' for i in range(0, components)]
    principal_df = pd.DataFrame(data=X_pca, columns=cols)
    return pd.concat([principal_df, y], axis=1)


def draw_pca_2d(df):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [0, 1]
    colors = ['r', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = df['y'] == target
        ax.scatter(df.loc[indicesToKeep, 'component 1'], 
                    df.loc[indicesToKeep, 'component 2'], 
                    c = color, s = 50)
    ax.legend(targets)
    ax.grid()
    return


from mpl_toolkits.mplot3d import Axes3D
def draw_pca_3d(df):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1, projection='3d') 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title('3 component PCA', fontsize = 20)
    targets = [0, 1]
    colors = ['r', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = df['y'] == target
        ax.scatter(df.loc[indicesToKeep, 'component 1'], 
                    df.loc[indicesToKeep, 'component 2'], 
                    df.loc[indicesToKeep, 'component 3'], 
                    c = color, s = 50)
    ax.legend(targets)
    ax.grid()
    return


def plot_2_param_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val), color=color)
        # add sd visualisation
        #ax.fill_between(grid_param_1, scores_mean[idx,:]-scores_sd[idx,:], \
        # scores_mean[idx,:]+scores_sd[idx,:], alpha=0.3, facecolor=color)

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    return


def plot_1_param_grid_search(cv_results, grid_param_1, name_param_1):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean)

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd)

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    ax.bar(grid_param_1, scores_mean, yerr=scores_sd)

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    return


def plot_all_gridsearch_results(grid):
    """
    Params: 
        grid: A trained GridSearchCV object.
    """
    ## Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    # means_train = results['mean_train_score']
    # stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params=grid.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        
        x = np.array(list(i[1] for i in params[p]) if any(isinstance(i, dict) for i in params[p]) else params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        # y_2 = np.array(means_train[best_index])
        # e_2 = np.array(stds_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        # ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train' )
        ax[i].set_xlabel(p.upper())
        ax[i].grid('on')

    plt.legend()
    plt.show()
    return




def calculate_vif(features: list, large_df: pd.DataFrame):
    # Compute VIF data for each independent variable (considered high if > 10)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    features.append('y')
    df_chosen = large_df[features]
    vif = pd.DataFrame()
    vif["features"] = df_chosen.columns
    vif["vif_Factor"] = [variance_inflation_factor(df_chosen.values, i) \
                        for i in range(df_chosen.shape[1])]
    features.remove('y')
    return vif