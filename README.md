# RawNeXt

Pytorch code for following paper:

* **Title** : RawNeXt: Speaker verification system for variable-duration utterance with deep layer aggregation and dynamic scaling policies
* **Autor** : Ju-ho Kim, Hye-jin Shim, Jungwoo Heo, and Ha-Jin Yu

# Abstract
<img align="middle" width="2000" src="https://github.com/wngh1187/RawNeXt/blob/main/overall.png">

Despite achieving satisfactory performance in speaker verification due to deep neural networks, the variable-duration utterance still remains a challenge that threatens the robustness of system. 
To deal with this issue, we propose a speaker verification system called ***RawNeXt*** that can handle input raw waveforms of arbitrary length by the following two components: 
(1) A deep layer aggregation strategy fuses speaker information of various time scales by aggregating stages and block iteratively and hierarchically. 
(2) A dynamic scaling policy selectively merges the activations of the various resolution branches in each block according to the length of the utterance. 
Owing to these two methods, our proposed model can extract speaker embeddings rich in time-frequency information by fully exploiting utterances and operating dynamically on length variation. 
Experimental results on the VoxCeleb1 test set of various length demonstrate that the RawNeXt achieve state-of-the-art performance compared to the existing short utterance speaker verification systems. 

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
