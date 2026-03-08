# ADNI3_machine_learning


Installation
---------------------------
The code has been tested and validates with Python `3.9`. Please, install the 
packages in the `requirements.txt` file with the corresponding Python version.


Usage
---------------------------
The repository only contains the ``all_classifiers_run_pipeline_Trophic_sc_gr_DK80.py`` runnable script. 
It contains a series of configurable parameters at the head of the script that can 
be modified directly on the code. 

When running the script with the selected parameters, the pipeline automatically 
generates the reports and results in an automatically created folder ``/Results``.

NOTES ON NEW RELEASE
----------------
mRMR mutual_info_classif added as Y is a binary label 
mRMR now is deterministic: random state added
mRMR is performed on the train set of each LOOC, before was on the global train set
Youden's J is now only calculated from the train set and the threshold applied to the test set
