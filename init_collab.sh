git init
git remote add -t \* -f origin https://github.com/jlbyoung/CMPT-414-Rotoscoping.git
git checkout investigate-loss-and-metrics # Replace with your branch here

# We can add all the commands we want to run here
python train.py -c collabConfig.json