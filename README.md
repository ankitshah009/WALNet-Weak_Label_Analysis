## WALNet Weak Label Analysis


This is an implementation for the paper "[A Closer Look at Weak Label Learning for Audio Events](https://arxiv.org/abs/1804.09288)". In this paper, we attempt to understand the challenges of large scale Audio Event Detection (AED) using weakly labeled data through a CNN based framework. Our network architecture is capable of handling variable length recordings and architecture design provides a way to control segment size of adjustable secondary outputs and thus these features eliminate the need for additional preprocessing steps. We look into how label density and label corruption affects performance and further compare mined web data as training data in comparison with manually labelled training data from AudioSet. We believe our work provides an approach to understand the challenges of weakly labeled learning and future AED works would benefit from our exploration. 

We provide the Audioset data (list of files used in our experimentation) provided for reproducibility. 

### WALNet Architecture Diagram

![WALNet Architecture Diagram](https://github.com/ankitshah009/WALNet-Weak_Label_Analysis/blob/master/WALNet_Architecture_DIagram.jpg)

### Web Page for More Details on Experimentation: - 

[Visit our Web Page here](https://ankitshah009.github.io/weak_label_learning_audio)

### Feature Level Representation - Download here

[128 dimensional MelSpectrogram Features - Balanced Set - 10 second](https://drive.google.com/file/d]/11FYyEX1xHqhR0b5Meoi4vqarjsaHMfSy/view?usp=sharing)

[128 dimensional MelSpectrogram Features - Validation Set - 10 second](https://drive.google.com/file/d/11G_Ii0IPUaq0d83BdibVoj7C3cfNjI4d/view?usp=sharing)

[128 dimensional MelSpectrogram Features - Testing Set - 10 second](https://drive.google.com/file/d/11GrJrjhWT4aieVgM0HogSwXHdwBnKPhg/view?usp=sharing)

[128 dimensional MelSpectrogram Features - Balanced Set - 30 second](https://drive.google.com/file/d/11Wb61gXWqsA2LimhVw89a8UDXMdQgp6Y/view?usp=sharing)

[128 dimensional MelSpectrogram Features - Balanced Set - 60 second](https://drive.google.com/file/d/1RCiO5rAA96_pBivP5qrCRbIR5eujNuau/view?usp=sharing)

### Embedding level feature representation - Download here

[All Embedding level features for AudioSet experiments](https://drive.google.com/file/d/1RFwssy3Ftz6YpYwAAiu7-vtczwfCAtmm/view?usp=sharing)

Reference
==========

<a href="https://arxiv.org/pdf/1804.09288.pdf"><img src="https://img.shields.io/badge/download%20paper-PDF-ff69b4.svg" alt="Download paper in PDF format" title="Download paper in PDF format" align="right" /></a>

If you use our repository or feature representation for your research WALNet- weak label analysis, please cite our paper:

    
	@article{shah2018closer,
  	title={A Closer Look at Weak Label Learning for Audio Events},
  	author={Shah, Ankit and Kumar, Anurag and Hauptmann, Alexander G and Raj, Bhiksha},
  	journal={arXiv preprint arXiv:1804.09288},
  	year={2018}
	}
    

Latest Results - 
===============
#### Use these numbers while reporting - arXiV paper update coming soon as version 2


| Training Set  | MAP on Testing	 |
| ------------- | ------------- |
| AudioSet - 10 | 22.87  |
| AudioSetAt30 | 22.42 |
| AudioSetAt60 | 22.42 |

 ##### Mean Average Precision on Audioset - 10 second recordings 

  | Model | MAP |
  | ----- | ---- |
  | ConvNet (mean pooling) | 20.3 |
  | ResNet (mean pooling) |  21.8 |
  | ResNet-ATT [Xu et al., 2017a] | 22.0 |
  | ResNet-SPDA [Zhang et al., 2016] | 21.9 |
  | Mmnet [Chou et al., 2018] | 22.6 |
  | WALNet [Shah.et.al, 2018] | 22.9 |



| ESC-50 dataset  | MAP	 |
| ------------- | ------------- |
| SoundNet | 74.2  |
| WALNet | 83.5 |

    
## Questions

Contact Ankit Shah (aps1@andrew.cmu.edu) or Anurag Kumar (alnu@andrew.cmu.edu)
