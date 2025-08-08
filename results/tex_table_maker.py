from pathlib import Path

import matplotlib.pyplot as plt

from results.constants import INFERENCE_METRIC, TEMPORAL_METRIC, ROBUSTNESS_METRIC, PERFORMANCE_METRIC, E2E_MODELS, \
    FEATURE_MODELS, CERTIFICATION_MODELS, IMG_MODELS
from results.inference.inference_metric import collect_training_time
from results.leaderboard import compute_benchmark


def _scatter_family_models(model_names, data, metric_a, metric_b, label):
    specific_models = data[data['model'].isin(model_names)]
    x1 = specific_models[metric_a]
    x2 = specific_models[metric_b]
    plt.scatter(x1, x2, label=label)


def scatter_metrics(data, metric_a, metric_b, title):
    plt.figure()
    _scatter_family_models(E2E_MODELS, data, metric_a, metric_b, "E2E")
    _scatter_family_models(FEATURE_MODELS, data, metric_a, metric_b, "phi")
    _scatter_family_models(IMG_MODELS, data, metric_a, metric_b, "img")
    _scatter_family_models(CERTIFICATION_MODELS, data, metric_a, metric_b, "cert")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel(metric_a)
    plt.ylabel(metric_b)
    plt.legend()
    plt.title(title)
    plt.show()


def produce_training_table(base_path):
    data = collect_training_time()
    tr_table_output = base_path / "training_time.tex"
    data.to_latex(tr_table_output,
                  header=["Model", "Time (h)"],
                  float_format="{:.2f}".format,
                  index=False)


def produce_tex_tables(limit_up: int = 5, limit_down: int = 5):
    data = compute_benchmark()
    base_path = Path(__file__).parent / "tables"
    base_path.mkdir(exist_ok=True)
    produce_training_table(base_path)
    extract_local_leaderboard(base_path, data, limit_up, limit_down)


def extract_local_leaderboard(base_path, data, limit_up: int = 5, limit_down: int = 5):
    table_output = base_path / "rank.tex"
    data = data[["model", "rank", PERFORMANCE_METRIC, TEMPORAL_METRIC, ROBUSTNESS_METRIC, INFERENCE_METRIC]]
    data.to_latex(table_output, header=["Model", "Rank", "P", "T", "R", "C"], float_format="{:.2f}".format, index=False)
    perf_table_output = base_path / "p_rank.tex"

    p_rank = data[["model", PERFORMANCE_METRIC]].sort_values(by=PERFORMANCE_METRIC, ascending=False)
    p_rank.to_latex(perf_table_output,
                    header=["Model",
                            "P"],
                    float_format="{:.2f}".format,
                    index=False)

    temp_table_output = base_path / "t_rank.tex"
    data[["model", TEMPORAL_METRIC]].sort_values(by=TEMPORAL_METRIC, ascending=False).to_latex(temp_table_output,
                                                                                               header=["Model", "T"],
                                                                                               float_format="{:.2f}".format,
                                                                                               index=False)
    rob_table_output = base_path / "r_rank.tex"
    data[["model", ROBUSTNESS_METRIC]].sort_values(by=ROBUSTNESS_METRIC, ascending=False).to_latex(rob_table_output,
                                                                                                   header=["Model",
                                                                                                           "R"],
                                                                                                   float_format="{:.2f}".format,
                                                                                                   index=False)
    inf_table_output = base_path / "c_rank.tex"
    data[["model", INFERENCE_METRIC]].sort_values(by=INFERENCE_METRIC, ascending=False).to_latex(inf_table_output,
                                                                                                 header=["Model", "C"],
                                                                                                 float_format="{:.2f}".format,
                                                                                                 index=False)

if __name__ == '__main__':
    produce_tex_tables()
