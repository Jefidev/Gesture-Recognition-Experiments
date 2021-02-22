# Sign Language Recognition Experiments

This repository contains the implementation of various models for video recognition. Those models were benchmarked on [the LSFB dataset](https://projects.info.unamur.be/lsfb-dataset/) depicting 395 sign from the Belgian Sign Languages. Other sign language dataset were alsos used for comparison (MS-ASL and GSL)


## Models
- **CNN + RNN** : Model computing the VGG16 embedding for each frame of the video and feeding them to a RNN networks in order to classify signs
- **C3D** : 3D convulition model for video recogniton cloned from https://github.com/csuhuihui/pytorch-c3d
- **I3D** : Inflated convolutionnal network cloned from https://github.com/piergiaj/pytorch-i3d

# Results

The accuracy obtained on the models are :

|               |  CNN + RNN |  C3D  |  I3D  |
|---------------|:----------:|:-----:|:-----:|
| LSFB-ISOL     |  3.6%      |  6.4% | 51%   |
| MSASL-100     |  0.8%      |  1.3% | 53%   |
| GSL           |  6.1%      |  8.6% | 36.5% |
