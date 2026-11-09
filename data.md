# Dataset Choice: VGGFace2

**Project:** Face Recognition Through Transfer Learning and Deep Metric Learning  
**Phase:** 1 ? Dataset Preparation and Face Identification Baseline  
**Framework:** Python, PyTorch, Torchvision  
**Selected dataset:** VGGFace2

## 1. What we want to learn

Our first task is **closed-set face identification**. Given a face image, the model must predict which person it shows from a fixed set of known people. Each person is one class.

We plan to compare a small CNN trained from scratch with an ImageNet-pretrained ResNet18. Later, we plan to study **face embeddings**: vectors that describe faces and allow us to compare images of people who were not used in training.

## 2. Why we chose VGGFace2

VGGFace2 was built for face recognition across changes in head pose and age. Its images also vary in lighting and appearance. The original dataset has about **3.31 million images of 9,131 people**, with an average of about **363 images per person**. Its official split has **8,631 training identities** and **500 test identities**. These figures describe the original release, not a verified count of our downloaded copy. [1]

This fits our project for three reasons:

- **Many examples of each person.** We can plan separate training, validation, and test images while keeping each selected person in all three sets.
- **Changes within the same identity.** The task can test whether a model learns identity features across different views and appearances.
- **Room for later work.** Separate identities let us plan embedding tests on people excluded from model training.

These are reasons for choosing the data. They do not prove that our models will perform well.

## 3. Why we changed from CelebA

The first report proposed CelebA. It includes **202,599 images of 10,177 people**, along with 40 face attributes and identity labels. It can support identity classification, but its main focus is face attributes. [2]

| Selection factor | CelebA | VGGFace2 |
|---|---|---|
| Main research focus | Face attributes | Face recognition across pose and age |
| Average images per person | About 20 | About 363 |
| Role in our project | Initial option | Selected dataset |

The averages come from the published image and identity counts. They do not mean every person has that many images. [1, 2]

We chose VGGFace2 because more examples per identity better support our planned classification splits and later embedding experiments. CelebA remains a valid option for a smaller study or for studying attributes.

## 4. How we plan to use it

The first experiment will use about **50 identities** from the official training partition. We will count the available images, choose a minimum image count, and select identities with a fixed random seed of **42**. Final sample counts will be reported after inspection.

For these known identities, we plan custom image splits:

| Split | Target share | Purpose |
|---|---|---|
| Training | 70% | Learn model parameters |
| Validation | 15% | Choose settings and the best checkpoint |
| Test | 15% | Measure final performance |

Each split will contain the same identities but different images. We will check exact and near duplicates before splitting. Related images should stay together to reduce **data leakage**, where very similar samples make test results look better than they should.

The official test identities will stay separate for later unseen-person experiments. They are not the test classes for the first closed-set classifier.

Planned preprocessing is RGB conversion, resizing to **128 ? 128**, and ImageNet normalization. Both models will use the same splits and input processing. Augmentation will apply only to training images. Validation and test processing will be fixed.

## 5. Downloading and current status

We use the [RichardErkhov/VGGFace2 Hugging Face mirror](https://huggingface.co/datasets/RichardErkhov/VGGFace2/tree/main/data), which provides image archives. Its training archive is about **37.9 GB**. Local storage must hold both the cached archive and extracted images. [3]

```powershell
python -m src.data.prepare
```

This command downloads and extracts training images under `data/raw/vggface2/train/`. Add `--include-test` to acquire the official test archive too.

**Only downloading and extraction are implemented in the data pipeline.** Image audits, identity selection, splits, preprocessing, and data loaders are planned. No dataset counts or model accuracy results have been measured yet.

## 6. Limits of this choice

A large dataset costs storage and preparation time. Using a subset will keep the first experiment manageable on our planned GPU server.

Celebrity images may not reflect the people or camera conditions of a real application. We must also inspect image quality, class imbalance, and possible label errors. Results on our selected identities will not establish performance for all people or settings.

A closed-set classifier cannot automatically recognize new people. That requires a separate embedding-based method and evaluation. We will review dataset usage terms and keep downloaded images out of Git.

## References

1. Cao, Q., Shen, L., Xie, W., Parkhi, O. M., and Zisserman, A. (2018). *VGGFace2: A Dataset for Recognising Faces Across Pose and Age*. FG. [Official project repository](https://github.com/ox-vgg/vgg_face2) and [paper](https://arxiv.org/abs/1710.08092).
2. Liu, Z., Luo, P., Wang, X., and Tang, X. (2015). *Deep Learning Face Attributes in the Wild*. ICCV. [Official CelebA page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
3. RichardErkhov. *VGGFace2 dataset mirror*. [Hugging Face archive listing](https://huggingface.co/datasets/RichardErkhov/VGGFace2/tree/main/data).
