# RawNeXt

Pytorch code for following paper:

* **Title** : RawNeXt: Speaker verification system for variable-duration utterance with deep layer aggregation and dynamic scaling policies [here]( https://arxiv.org/abs/2112.07935 ) 
* **Autor** : Ju-ho Kim, Hye-jin Shim, Jungwoo Heo, and Ha-Jin Yu

# Abstract
<img align="middle" width="2000" src="https://github.com/wngh1187/RawNeXt/blob/main/overall.png">

Despite achieving satisfactory performance in speaker verification using deep neural networks, variable-duration utterances remain a challenge that threatens the robustness of systems. 
To deal with this issue, we propose a speaker verification system called RawNeXt that can handle input raw waveforms of arbitrary length by employing the following two components: 
(1) A deep layer aggregation strategy enhances speaker information by iteratively and hierarchically aggregating features of various time scales and spectral channels output from blocks. 
(2) An extended dynamic scaling policy flexibly processes features according to the length of the utterance by selectively merging the activations of different resolution branches in each block. 
Owing to these two components, our proposed model can extract speaker embeddings rich in time-spectral information and operate dynamically on length variations. 
Experimental results on the VoxCeleb1 test set consisting of various duration utterances demonstrate that RawNeXt achieves state-of-the-art performance compared to the recently proposed systems. 

# Prerequisites

## Environment Setting
* We used 'nvcr.io/nvidia/pytorch:21.04-py3' image of Nvidia GPU Cloud for conducting our experiments. 
* We used four Quadro rtx-5000 GPUs for training. 
* Python 3.6.9
* Pytorch 1.8.1
* Torchaudio 0.8.1

## Datasets

We used VoxCeleb2 dataset for training and VoxCeleb1 dataset for test with three evaluation trials. 
For data augmentation, we used room impulse response simulation and MUSAN corpus. 
We referenced the data augementation code at [here]( https://github.com/clovaai/voxceleb_trainer )


# Training

```
Go into run directory
Activate the code you want in train.sh
./train.sh
```

# Test

```
Go into run directory
Activate the code you want in test.sh
./test.sh
```

# Citation
Please cite this paper if you make use of the code. 

```
@article{kim2021rawnext,
      title={RawNeXt: Speaker verification system for variable-duration utterances with deep layer aggregation and extended dynamic scaling policies}, 
      author={Ju-ho Kim and Hye-jin Shim and Jungwoo Heo and Ha-Jin Yu},
      year={2021},
      eprint={2112.07935},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
