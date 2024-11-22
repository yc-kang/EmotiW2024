# Multimodal Engagement Classification - EmotiW2024

## Introduction
ACM [EmotiW2024](https://sites.google.com/view/emotiw2024/home) challenge, we focused the subchallenge: Engagement classification on videos.

## Dataset and baselines
We worked with [EngageNet](https://github.com/engagenet/engagenet_baselines), with a pre-ensemble baseline.

<div style="vertical-align:middle"><img src="images/figure1.png" alt="Dataset" width="550px" text-align="center">
</div>

## Architecture
The model is ensembled from: Pose Tracking, Facial Landmarks, Facial Features, Video Understanding

<div style="text-align:center"><img src="images/figure2.png" alt="Model Architecture" width="550px" align="center">
</div>

## Code Layout
Structure: [here](Directory_Structure.md)
- notebooks/augmentation - Data augmentation
- notebooks/preprocessing - Data preprocessing pipelines
- notebooks/ensemble - Model ensemble from different modalities

## Results
### Individual Modalities
Based on EngageNet Validation Set
| Modality  | Accuracy | F1-Score
| ------------- | ------------- | ------------- |
| Pose  | 0.654 | 0.60 |
| Landmark  | 0.4889 | 0.40 |
| Face | 0.7087 | 0.60 |
| Video Understanding | - | - |

### Ensembling Performance
| Ensemble  | Accuracy | F1-Score
| ------------- | ------------- | ------------- |
| Transformer  | - | - |
| Transformer-Fusion  | - | - |
| Model x  | - | - |
| Model y | - | - |
| Model z | - | - |

### Ablation Study
| Ensemble  | Accuracy | F1-Score
| ------------- | ------------- | ------------- |
| Pose-Land-Face  | - | - |
| Pose-Land-Vid  | - | - |
| Pose-Face-Vid  | - | - |
| Land-Face-Vid | - | - |

### Table - Final Ensemble
| Dataset  | Accuracy
| ------------- | -------------
| Validation | **-** |
| Test | **-** |

## The Team
Yichen Kang, Yanchun Zhang, Jun Wu  
[EESM5900V - HKUST](https://cqf.io/EESM5900V/)  
The Hong Kong University of Science and Technology (HKUST)
