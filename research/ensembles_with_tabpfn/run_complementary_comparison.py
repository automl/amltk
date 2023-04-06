from research.ensembles_with_tabpfn.complementary_code.data_handler import read_all_base_models
from research.ensembles_with_tabpfn.utils.config import ALGO_NAMES

from byop.ensembling.ensemble_preprocessing import prune_base_models


def _run():

    path_to_base_model_data = "./base_model_data"
    bm_bucket_name = "debug"

    # -- Get all base models
    base_models = read_all_base_models(path_to_base_model_data, bm_bucket_name, ALGO_NAMES)

    # - Get the best model per algorithm (based on validation score)
    base_models = prune_base_models(base_models, max_number_base_models=len(ALGO_NAMES), pruning_method="SiloTopN")

    print()


if __name__ == "__main__":
    # TODO:
    #   - make agnostic to dataset
    _run()
