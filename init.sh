# Sometimes the notebook may run into random issues, if that happens just run it again

# Setup repository
git init
git remote add -t \* -f origin https://github.com/jlbyoung/CMPT-414-Rotoscoping.git
git checkout master # Replace with your branch here

#install packages
conda env create -f environment.yml
conda activate cv414
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch


# We can add all the commands we want to run here
python train.py -c collabConfig.json