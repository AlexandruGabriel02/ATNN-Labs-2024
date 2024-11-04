# ATNN Homework 3

Report: [report.MD](report.MD).

To run a pipeline on a specific configuration: `python pipeline.py <pipeline_config_file.json>`

To run a parameter sweep, first define your general pipeline configuration in a json config file like the one above.
Then, specify the desired parameters in another sweep config json file.
Afterwards, run the following: `python sweep.py <sweep_parameters.json> <pipeling_config_file.json>`
