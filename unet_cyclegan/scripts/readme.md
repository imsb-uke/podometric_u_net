# About
Folder for scripts using U-Net CycleGAN. 

## Hyperparameters
Hyperparameters are set using sacred (.yaml config file). Which files are used
for training, validation and testing has to be defined via CSV files that have to start with "train", 
"validate" and "test". 

## Important assumptions
Suffixes of CSV files can be set via the config file. E.g. if "csv_suffix" is set to "_kidney", 
the files "train_kidney.csv", "validate_kidney.csv" and "test_kidney.csv" would be used. 
The CSV files have to be located in the lowest folder of the used data and are set via "dataset_lowest_folder" 
in the config file. If this is unwanted, the example script can be adapted.
