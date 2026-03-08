import json
import multiprocessing

from pathlib import Path

from src.data_loading.load_from_excel import get_x_arr_for_scikit_from_excel
from src.pipelines.train_eval_pipelines import (
    train_eval_gridsearch_loocv_with_outer_n_loop,
)

# Determine the number of CPU cores to use
n_jobs = multiprocessing.cpu_count() - 1  # Leave one core free

data_files = {
    "all_features_TL_combat": "ML_Trophic_ADNI3_4STAGINGBYABETA_N134_DK80_sc_gr.xlsx",
}

classifier = "LogReg"
classifications = ["HCneg_vs_ADpos", "HCneg_vs_MCIpos", "MCIpos_vs_ADpos","HCneg_vs_HCpos"]
group_labels = {"HCneg": "HC_ABneg", "HCpos": "HC_ABpos","MCIpos": "MCI_ABpos", "ADpos": "AD_ABpos"}

if __name__ == "__main__":
    # Main execution
    path_repo = Path(Path(__file__).parent / ".." / "..").resolve()
    excel_folder = path_repo / "Data" / "trophic"
    param_folder = path_repo / "Parameters"
    for classification in classifications:
        for data_type in data_files.keys():
            print(f"Training {classifier} with parameters_{classifier}.json")

            results_folder = (
                    path_repo / "Results" / "final_3d_gs_classification_trophic_DK80_sc_gr_det_run2" /
                    classifier / classification / data_type
            )
            results_folder.mkdir(exist_ok=True, parents=True)

            data_file = excel_folder / data_files[data_type]

            # Load the parameters file
            param_file = path_repo / "Parameters" / f"parameters_{classifier}.json"
            with open(param_file, "r") as file:
                parameters = json.load(file)
            parameters["N_JOBS"] = n_jobs

            if classification == "HCneg_vs_HCpos":
                parameters["GROUPS"] = {"HC_ABneg": 0, "HC_ABpos": 1}
            elif classification == "HCneg_vs_MCIpos":
                parameters["GROUPS"] = {"HC_ABneg": 0, "MCI_ABpos": 1}
            elif classification == "HCneg_vs_ADpos":
                parameters["GROUPS"] = {"HC_ABneg": 0, "AD_ABpos": 1}
            elif classification == "MCIpos_vs_ADpos":
                parameters["GROUPS"] = {"MCI_ABpos": 0, "AD_ABpos": 1}
            else:
                pass

            print(f"Training N fold with {classifier} Grid Search and LOOCV {data_file} Data")

            # We load the data from the file
            x, y, feature_columns = get_x_arr_for_scikit_from_excel(
                data_file,
                parameters["GROUPS"],
                parameters["ID_KEY"],
                parameters["GROUP_KEY"],
            )

            # And we run the train_eval_knn gridsearch which performs gridsearch,
            # trains on best parameters, and evaluates the model and reports the
            # performances.
            train_eval_gridsearch_loocv_with_outer_n_loop(
                x, y, feature_columns, parameters, results_folder,
                n=20
            )
