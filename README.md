# CMPT-414-Rotoscoping
---

Rotoscoping Project
James Young, Ahad Chaudhry, Julian Laxman
jlyoung@sfu.ca, ahadc@sfu.ca ,jlaxmane@sfu.ca



<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

- [# CMPT-414-Rotoscoping](#h1-id%22cmpt-414-rotoscoping-24%22cmpt-414-rotoscopingh1)
- [Requirements](#requirements)
- [Environment](#environment)
  - [Setup](#setup)
- [Folder Structure](#folder-structure)
- [Usage](#usage)
  - [Config file format](#config-file-format)
  - [Using config files](#using-config-files)
  - [Resuming from checkpoints](#resuming-from-checkpoints)
  - [Using Multiple GPU](#using-multiple-gpu)
- [TODOs](#todos)
- [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* Python == 3.7 
* PyTorch == 1.4 
* tqdm (Optional for `test.py`)
* tensorboard == 1.15 (see [Tensorboard Visualization](#tensorboard-visualization))

## Environment  

### Setup

Recommended to have the Anaconda Distribution installed on your device, it will contain conda and many other packages for data science. Please also ensure that `conda` command is in your `$PATH` and that it works in your terminal.

```sh
conda env create -f environment.yml
conda activate cv414

conda install pytorch torchvision -c pytorch <device specific installation version>
```

## Folder Structure

  TODO: Update when project files are complete
  ```
  cmpt-414-rotoscoping/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
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
The code in this repo is an MNIST example of the template. TODO: Change this

Use `python train.py -c config.json` to train model. 

Use `python test.py -c config.json --resume <checkpointPath>` to test the model.

Use `tensorboard --logdir saved/log` to run tensorboard.


### Config file format
Config files are in `.json` format:
```javascript
{
  "name": "Mnist_LeNet",        // training session name
  "n_gpu": 1,                   // number of GPUs to use for training.
  
  "arch": {
    "type": "MnistModel",       // name of model architecture to train
    "args": {

    }                
  },
  "data_loader": {
    "type": "MnistDataLoader",         // selecting data loader
    "args":{
      "data_dir": "data/",             // dataset path
      "batch_size": 64,                // batch size
      "shuffle": true,                 // shuffle training data before splitting
      "validation_split": 0.1          // size of validation dataset. float(portion) or int(number of samples)
      "num_workers": 2,                // number of cpu processes to be used for data loading
    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 0.001,                     // learning rate
      "weight_decay": 0,               // (optional) weight decay
      "amsgrad": true
    }
  },
  "loss": "nll_loss",                  // loss
  "metrics": [
    "accuracy", "top_k_acc"            // list of metrics to evaluate
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

## TODOs

- [ ] New Task

## Acknowledgements
TODO: add link for template


