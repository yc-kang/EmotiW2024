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
Structure: 
- notebooks/augmentation - Data augmentation
- notebooks/preprocessing - Data preprocessing pipelines
- notebooks/ensemble - Model ensemble from different modalities

## Results
### Individual Modalities
Based on EngageNet Test Set
| Modality  | Accuracy | F1-Score
| ------------- | ------------- | ------------- |
| Pose | 0.698 | 0.69 |
| Landmark | 0.614 | 0.58 |
| Face | 0.689 | 0.67 |
| Video Understanding | 0.652 | 0.61 |

### Ensembling Performance
| Ensemble  | Accuracy
| ------------- | ------------- |
| Late-Fusion (Hard) | 0.676 |
| Late-Fusion (Soft) | 0.718 |
| Late-Fusion (Weighted) | 0.694 |
| Early-Fusion (Transformer Fusion) | 0.744 |

### Ablation Results
| Ensemble  | Accuracy
| ------------- | ------------- |
| Pose-Land-Face  | 0.743 |
| Pose-Land-Vid  | 0.740 |
| Pose-Face-Vid  | 0.747 |
| Land-Face-Vid | 0.695 |

### Table - Final Ensemble
| Dataset  | Accuracy
| ------------- | -------------
| Validation | 0.713 |
| Test | 0.747 |

## The Team
Yichen Kang, Yanchun Zhang, Jun Wu  
[EESM5900V - HKUST](https://cqf.io/EESM5900V/)  
The Hong Kong University of Science and Technology (HKUST)
