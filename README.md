Code repo for the SLAM project. 

Kevin O'Brien,
Mayank Singal,
Sanil Pande

First, install CARLA.
Then create a project folder
Inside the project folder, clone the repo https://github.com/magicleap/SuperGluePretrainedNetwork
Then clone this repo in the project folder.

# superglue_vio

## Directory Structure
```
<project folder>
	SuperGluePretrainedNetwork
		...
		...
	superglue_vio (this repo)
```

## Run Instructions
1. Launch Carla
```
cd <carla_dir>
./CarlaUE4.sh
```

2. Run script
```
cd superglue_vio
python vio_clean.py
```
