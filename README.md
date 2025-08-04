# Personalized Feature Translation for Expression Recognition: An Efficient Source-Free Domain Adaptation Method

<p align="center"><img src="promo.png" alt="outline" width="90%"></p>

## Abstract
Facial expression recognition (FER) models are employed in many video-based affective computing applications, such as human-computer interaction and healthcare monitoring.  However, deep FER models often struggle with subtle expressions and high inter-subject variability, limiting their performance in real-world applications. To improve their performance, source-free domain adaptation (SFDA) methods have been proposed to personalize a pretrained source model using only unlabeled target domain data, thereby avoiding data privacy, storage, and transmission constraints. This paper addresses a challenging scenario, where source data is unavailable for adaptation, and only unlabeled target data consisting solely of neutral expressions is available. SFDA methods are not typically designed to adapt using target data from only a single class. Further, using models to generate facial images with non-neutral expressions can be unstable and computationally intensive. 
%
In this paper, personalized feature translation (PFT) is proposed for SFDA. Unlike current image translation methods for SFDA, our lightweight method operates in the latent space. We first pre-train the translator on the source domain data to transform the subject-specific style features from one source subject into another. Expression information is preserved by optimizing a combination of expression consistency and style-aware objectives. Then, the translator is adapted on neutral target data, without using source data or image synthesis. By translating in the latent space, PFT avoids the complexity and noise of face expression generation, producing discriminative embeddings optimized for classification. Using PFT eliminates the need for image synthesis, reduces computational overhead (using a lightweight translator), and only adapts part of the model, making the method efficient compared to image-based translation. Extensive experiments on four challenging video FER benchmark datasets, BioVid, StressID, BAH, and Aff-Wild2, show that PFT consistently outperforms state-of-the-art SFDA methods, providing a cost-effective approach that is suitable for real-world, privacy-sensitive FER applications. 

## Installation of the environments
```bash
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
scikit-learn>=1.1.0
tensorboard>=2.10.0
argparse
opencv-python
dlib
scipy
tqdm
```


## BioVid database
```sh
Biovid datasets PartA can be downloaded from here: (https://www.nit.ovgu.de/BioVid.html#PubACII17)
```
## Dataset Structure & Pairing
The source training data should be organized into subject-specific folders. Each folder contains images with expression labels embedded in the filenames. The expression label (e.g., N for neutral, P for pain) appears at the end of each filename before the extension.
```sh
source_sub1/
├── Image1_P.jpg
├── Image2_N.jpg
...

source_sub2/
├── Image1_N.jpg
├── Image2_P.jpg
...
```

## Train the model on source domain
```sh
python main_src.py --epoch 100 --batchsize 20 --lr 1e-5
```

## Adaptation to target domains (subjects)
```sh
python main_tar.py --epoch 25 --batchsize 32 --lr 1e-4 --biovid_annot_train $Path to the training data --biovid_annot_val $Path to the validation data --save_dir $Directory to save experiment results --img_dir Directory to save generated images --par_dir Directory to save the best parameters
```
## Test
```sh
python test.py
```
