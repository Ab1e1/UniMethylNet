# UniMethylNet

This repository provides the implementation of ‘**UniMethylNet**, a universal prediction model for DNA methylation site analysis based on deep neural networks and attention mechanisms’. All necessary code, data processing scripts, and configuration files are included to help you reproduce our results and experiments. 

## Model Introduction

UniMethylNet is a universal DNA methylation site prediction model that integrates a Position-wise Linear Layer, a Bidirectional Long Short-Term Memory network, and a Channel-Spatial Dual Attention module. The Position-wise Linear Layer captures precise local sequence patterns, while the BiLSTM models long-range dependencies. The dual attention mechanism adaptively highlights informative regions, enabling the model to effectively extract methylation-related features across multiple scales and contexts.

## Usage

### Version Dependencies

    Python version:  3.7.10
    numpy==1.20.3
    torch==1.8.1+cu111
    scikit-learn==0.24.2

### Make sure all required packages are installed before executing the model. Use the commands below to set up the environment.

```
# Install PyTorch (CUDA 11.1 support)
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install numpy
pip install numpy==1.20.3

# Install scikit-learn
pip install scikit-learn==0.24.2

```

## Experimental Results

Experiments on multi-species methylation datasets show that UniMethylNet performs robustly across different methylation types. Detailed results can be found in the paper.

##  Get started

After completing the setup, you can start training by running `main_train.py`.The following arguments are required:

- model_type: cnn-rnn
- pos_fa/neg_fa: input files for positive samples/negative samples 
- out_dir: the path of output directory

Below is an example command for running the script:

```
python main_train.py -m model_type -pos_fa pos_fa -neg_fa neg_fa -od out_dir
```

