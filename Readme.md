# Generalizing Gaze Estimation with Weak-Supervision from Synthetic Views ([arxiv](https://arxiv.org/abs/2212.02997))


## Abstract

Developing gaze estimation models that generalize well to unseen domains and in-the-wild conditions remains a challenge with no known best solution. This is mostly due to the difficulty of acquiring ground truth data that cover the distribution of possible faces, head poses, and environmental conditions that exist in the real world. In this work, we propose to train general gaze estimation models based on 3D geometry-aware gaze pseudo-annotations which we extract from arbitrary unlabelled face images, which are abundantly available on the internet. Additionally, we leverage the observation that head, body, and hand pose estimation benefit from revising them as dense 3D coordinate prediction, and similarly express gaze estimation as regression of dense 3D eye meshes. We overcome the absence of compatible ground truth by fitting rigid 3D eyeballs on existing gaze datasets and designing a multi-view supervision framework to balance the effect of pseudo-labels during training. We test our method in the task of gaze generalization, in which we demonstrate improvement of up to $30\%$ compared to state-of-the-art when no ground truth data are available, and up to $10\%$ when they are.

![Teaser video](https://drive.google.com/file/d/17FoKC9omApzDri4XxouksqHK17N2HTk5/view?usp=sharing)

