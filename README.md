# RawNeXt

Pytorch code for following paper:

* **Title** : RawNeXt: Speaker verification system for variable-duration utterance with deep layer aggregation and dynamic scaling policies (The paper will be uploaded later) 
* **Autor** : Ju-ho Kim, Hye-jin Shim, Jungwoo Heo, and Ha-Jin Yu

# Abstract
<img align="middle" width="2000" src="https://github.com/wngh1187/RawNeXt/blob/main/overall.png">

Despite achieving satisfactory performances in speaker verification due to deep neural networks, the variable-duration utterance still remains a challenge that threatens the robustness of system. 
To deal with this issue, we propose a speaker verification system called ***RawNeXt*** that can handle input raw waveforms of arbitrary length by the following two components: 
(1) A deep layer aggregation strategy enhances speaker information by iteratively and hierarchically aggregating features of various time scales and frequency channels output from stages and blocks. 
(2) A dynamic scaling policy flexibly manipulates features according to the length of the utterance by selectively merging the activations of different resolution branches in each block. 
Owing to these two methods, our proposed model can extract speaker embeddings rich in time-frequency information and operate dynamically on length variation. 
Experimental results on the VoxCeleb1 test set of various duration demonstrate that the RawNeXt achieves state-of-the-art performance compared to the recently proposed systems. 

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
```
To be uploaded
```
