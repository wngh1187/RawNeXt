# RawNeXt: Speaker verification system for variable-duration utterance with deep layer aggregation and dynamic scaling policies

Pytorch code for following paper:

* **Title** : RawNeXt: Speaker verification system for variable-duration utterance with deep layer aggregation and dynamic scaling policies
* **Autor** : Ju-ho Kim, Jungwoo Heo, and Ha-Jin Yu

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
We used Nvidia GPU Cloud for conducting our experiments. We used the 'nvcr.io/nvidia/pytorch:21.04-py3' image. Refer to *launch_ngc.sh*. We used four Quadro rtx-5000 GPUs for training. 

## Datasets

We used VoxCeleb2 dataset for training and VoxCeleb1 dataset for test with three evaluation trials. 


```
예시
```

# Training

아래 사항들로 현 프로젝트에 관한 모듈들을 설치할 수 있습니다.

```
Go into run directory
./run.sh
```

# Test

어떻게 테스트가 이 시스템에서 돌아가는지에 대한 설명을 합니다

### 테스트는 이런 식으로 동작합니다

왜 이렇게 동작하는지, 설명합니다

```
예시
```

# Citation
Please cite the following if you make use of the code.


```
예시
```
