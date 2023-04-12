from byop.store import PathBucket

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Global font size update
plt.rcParams['font.size'] = '16'


def _corr_matrix_plot(corr_matrix, complement_model_name):
    # re-order to have complement model last
    col_order = list(corr_matrix)
    col_order.remove(complement_model_name)
    corr_matrix = corr_matrix[col_order + [complement_model_name]]

    # Make it not squared such that bottom row and first column are gone.
    masked_corr = corr_matrix.loc[~np.all(corr_matrix == -1, axis=1), ~np.all(corr_matrix == -1, axis=0)]

    # -- Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(masked_corr, annot=True, mask=masked_corr == -1)
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()

    # -- Make complement thing stand out somehow
    highlight_x_ticks = [x for x in ax.get_xticklabels() if x.get_text() == complement_model_name][0]
    x_cord = highlight_x_ticks._x
    # set the properties of the ticklabel
    highlight_x_ticks.set_weight("bold")
    highlight_x_ticks.set_size(20)
    highlight_x_ticks.set_color("green")

    for annot in ax.texts:
        if annot._x == x_cord:
            # set the properties of the heatmap annot
            annot.set_weight("bold")
            annot.set_color("green")
            annot.set_size(20)

    plt.show()


def _run():
    path_to_analysis_data = "./data_space/analysis_data"
    bm_bucket_name = "debug"

    result_bucket = PathBucket(path_to_analysis_data + f"/{bm_bucket_name}")
    result_stats = result_bucket["results_stats.json"].load()
    all_base_models_correlation_df = result_bucket["correlation_matrix.csv"].load()

    _corr_matrix_plot(all_base_models_correlation_df, "XT")

    print(result_stats)


if __name__ == "__main__":
    _run()
