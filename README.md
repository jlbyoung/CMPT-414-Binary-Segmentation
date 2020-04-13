# CMPT-414-Binary-Segmentation
---

James Young, Ahad Chaudhry, Julian Laxman <br/>
jlyoung@sfu.ca, ahadc@sfu.ca ,jlaxmane@sfu.ca



<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

- [CMPT-414-Binary-Segmentation](#cmpt-414-binary-segmentation)
  - [Requirements](#requirements)
  - [Environment](#environment)
    - [Setup](#setup)
  - [Folder Structure](#folder-structure)
  - [Usage](#usage)
    - [Config file format](#config-file-format)
    - [Using config files](#using-config-files)
    - [Resuming from checkpoints](#resuming-from-checkpoints)
    - [Using Multiple GPU](#using-multiple-gpu)
  - [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* Python == 3.7 
* PyTorch == 1.4 
* tqdm (Optional for `test.py`)
* tensorboard == 1.15 (see [Tensorboard Visualization](#tensorboard-visualization))
* albumentations == 0.4.5

## Environment  

### Setup

Please do not clone and use `init.sh` instead. Recommended to have the Anaconda Distribution installed on your device, it will contain conda and many other packages for data science. Please also ensure that `conda` command is in your `$PATH` and that it works in your terminal. Note the project is configured to work on UNIX based operating systems and expects to have a NVIDIA GPU (required to make the model run without issues). Cuda tools version should be 10.1.

```sh

# This may take a while (expected time is 15 minutes), please enter y when required
chmod +x init.sh && ./init.sh
cd CMPT-414-Binary-Segmentation
conda activate cv414
```

## Folder Structure

  ```
  CMPT-414-Binary-Segmentation/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  ├── webcam.py - evaluate model_best.pth using your webcam
  ├── demo_model.ipynb - demo model_best.pth on the demo.jpeg image
  ├── demo.jpeg - demo image
  ├── init.sh - Used to initialize the project
  ├── download_dataset.sh - Used to download the dataset
  ├── model_best.pth - Best model (pre-trained)
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   ├── base_trainer.py
  │   └── base_dataset.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── notebooks/ - Notebooks 
  │   └── make_person_segmentation_set.ipynb - used for making person_train.txt and person_val.txt
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   ├── loss_implementations.py
  │   ├── enet.py
  │   ├── unet.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```

## Usage
The code in this repo is used to train semantic segmentation models. 

Use `python webcam.py` to preview the pretrained model. 

Use `python train.py -c config.json` to train model. 

Use `python test.py -c config.json --resume <checkpointPath>` to test the model.

Use `tensorboard --logdir saved/log` to run tensorboard.


### Config file format
Config files are in `.json` format:
```javascript
{
  "name": "Binary-Segmentation",        // training session name
  "n_gpu": 1,                   // number of GPUs to use for training.
  
  "arch": {
    "type": "ENet",       // name of model architecture to train
    "args": {

    }                
  },
  "data_loader": {
    "type": "VOCDataLoader",         // selecting data loader
    "args":{
      "data_dir": "data/",             // dataset path
      "batch_size": 4,                // batch size
      "shuffle": true,                 // shuffle training data before splitting
      "validation_split": 0.1          // size of validation dataset. float(portion) or int(number of samples)
      "num_workers": 4,                // number of cpu processes to be used for data loading
    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 5e-4,                     // learning rate
      "weight_decay": 2e-4,               // (optional) weight decay
      "amsgrad": true
    }
  },
  "loss": "binary_cross_entropy_loss",               // loss
  "metrics": [
    "binary_iou"            // list of metrics to evaluate
  ],                         
  "lr_scheduler": {
    "type": "StepLR",                  // learning rate scheduler
    "args":{
      "step_size": 50,          
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 100,                     // number of training epochs
    "save_dir": "saved/",              // checkpoints are saved in save_dir/models/name
    "save_freq": 1,                    // save checkpoints every save_freq epochs
    "verbosity": 2,                    // 0: quiet, 1: per epoch, 2: full
  
    "monitor": "min val_loss"          // mode and metric for model performance monitoring. set 'off' to disable.
    "early_stop": 10	                 // number of epochs to wait before early stop. set 0 to disable.
  
    "tensorboard": true,               // enable tensorboard visualization
  }
}
```

Add addional configurations if you need.

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

### Using Multiple GPU
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py --device 2,3 -c config.json
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```


## Acknowledgements

[Lovasz Softmax](https://github.com/bermanmaxim/LovaszSoftmax)

[Model Implementations](https://github.com/yassouali/pytorch_segmentation)

[Repository Template](https://github.com/victoresque/pytorch-template/)

[Binary IOU Metric](https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy)
