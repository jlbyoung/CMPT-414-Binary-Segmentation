# Sometimes the notebook may run into random issues, if that happens just run it again

 Setup repository
git clone https://github.com/jlbyoung/CMPT-414-Rotoscoping.git
cd CMPT-414-Rotoscoping


git config --global user.email ""
git config --global user.name ""


git checkout master # Replace with your branch here


#install packages
sudo chmod +rw /opt/conda/pkgs/qt-5.12.5-hd8c4c69_1/info/paths.json
conda env create -f environment.yml
conda install --name cv414 pytorch torchvision cudatoolkit=10.1 -c pytorchxw

chmod +x download_dataset.sh
./download_dataset.sh

cp imagesets/*.txt data/VOCdevkit/VOC2012/ImageSets/Segmentation
# We can add all the commands we want to run here
# python train.py -c collabConfig.json