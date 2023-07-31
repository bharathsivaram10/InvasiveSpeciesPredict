# InvasiveSpeciesPredict
## About
This is the final project for CSCI8980-Spatial Enabled AI at the University of Minnesota-Twin Cities for Spring 2022.
Authors: Bharath Sivaram (Neural Net code + paper) & Pranav Julakanti (RandomForest)

![image](https://github.com/bharathsivaram10/InvasiveSpeciesPredict/assets/20588623/97f66e97-d303-4dfb-9e28-e27d56298e05)


## Background & Objective
Invasive species cause havoc to the ecosystem, thereby creating issues in local economies as well. The impact can be mitigated by early 
identification and treatment of areas prone to these species. 
In this case, we use Minnesota DNR observation data to train a neural net and predict the dominating invasive species in a buffer zone, during a certain season.

## Approach
1) Use GPKG from [MN DNR](https://gisdata.mn.gov/dataset/env-invasive-terrestrial-obs) which has observations from 2016 to 2021 and includes features such as habitat, body type, and treatment status
2) Split up data by the 4 seasons to account for climate/precipitation factors
3) Use OSM to create buffers and find geo-feature vectors
4) Test two different network architectures (Adam and SGD Optimizers)

## Architectures and Results

![image](https://github.com/bharathsivaram10/InvasiveSpeciesPredict/assets/20588623/fd0546ab-e11d-4533-817b-ec2711b821a8)

![image](https://github.com/bharathsivaram10/InvasiveSpeciesPredict/assets/20588623/9119d919-210d-44d4-9a89-af0b51950b22)


## How to run

1) Clone repo
2) Download data file from using drive [link](https://drive.google.com/file/d/1r7P_eD6ybbM8muIdZ6kX65VODOBcExNY/view?usp=sharing)
3) Run predict.py
4) The preprocess.py gives information on how the information was extracted for network use
