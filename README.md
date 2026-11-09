# Face identification from images

University deep learning project comparing a small custom CNN trained from scratch with an ImageNet-pretrained ResNet18 adapted to a fixed set of identities.

## Status and scope

ResNet18 model construction and targeted offline technical checks are implemented. Dataset preparation, the custom CNN, the full training/evaluation workflow, and results are pending. No models have been trained on our identities and no demonstration is available yet.

This README is the current scope specification. It supersedes the conflicting requirements, architecture, schedule, and deliverables in `face_recognition_tracking_project_specification.pdf`. That original reference is preserved unchanged. Video tracking, video exports, and unknown-person rejection are optional future extensions, not first-phase requirements. No video package is included.

The first phase takes a face image and predicts an identity among the people represented in training. Face detection locates and crops faces; it is a supporting preprocessing step. Face identification predicts identity. Video tracking follows faces across frames.

A fixed-identity classifier cannot automatically recognize newly enrolled people without further training. Unknown-person rejection may be added later with separate known/unknown validation and test data and a properly evaluated threshold. Softmax confidence alone does not reliably detect unknown people. Initial predictions are closed-set classifications.

## Proposed comparison

- A small custom CNN trained from scratch, producing class logits for cross-entropy loss.
- An ImageNet-pretrained ResNet18 with a replacement head: train the head with the backbone frozen, then unfreeze selected layers and fine-tune at a lower learning rate.

Use the same dataset partitions, identity mapping, evaluation preprocessing, and comparable training budgets. Select settings using validation data only. Planned evaluation includes accuracy, macro precision/recall/F1, a confusion matrix, per-class errors, and training/validation curves. These are planned outputs, not existing results.

## Data protocol

Select a licensed identity-labeled dataset with enough independent images per person. Dataset selection and acquisition are pending. Obtain consent for participant images and keep sensitive images, names, embeddings, and identifying paths local. Avoid including private information in notebooks, logs, figures, or reports.

Known identities must appear in training, validation, and test sets using different images. Start with approximately 70%/15%/15% per identity, adjusting counts to retain every identity in every split. Keep exact and near-duplicates together and, where relevant, separate recording sessions across splits. If there are too few independent groups for an identity, collect more data or exclude it instead of allowing leakage.

Create reproducible splits with a fixed seed, audit them, and reuse them for both models. Keep split manifests and identity mappings local initially because paths and labels may identify participants. Fit any learned preprocessing on training data only. Reserve test data for final evaluation after model selection; never tune against it.

## Directory structure

```text
configs/                  Starting YAML configurations
data/raw/                 Local source images
data/processed/           Local prepared face images
data/splits/              Local split manifests and identity mapping
notebooks/                Data exploration and results-analysis outlines
src/data/                 Preparation, datasets/loaders, and transforms
src/models/               ResNet18 implementation and custom CNN placeholder
tests/                    Targeted offline ResNet18 technical checks
src/train.py              Shared training and validation (pending)
src/evaluate.py           Held-out metrics and error analysis (pending)
src/predict.py            Saved-checkpoint image inference (pending)
src/utils.py              Configuration, seeds, and device selection (pending)
app/gradio_app.py         Future image-upload demonstration
checkpoints/              Local model artifacts
outputs/                  Local generated outputs, including inference outputs
reports/figures/          Future reviewed figures
```

Empty artifact directories contain tracked `.gitkeep` files. Dataset contents, split manifests, checkpoints, and generated outputs are ignored.

## Environment setup

When ready to implement, use a supported Python 3 environment. From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux activate with `source .venv/bin/activate`. Requirements are intentionally unpinned starting dependencies; record tested versions when implementation begins. Choose an appropriate PyTorch build for the eventual machine using its official installation guidance; no hardware-specific build is prescribed. This scaffold installs or downloads nothing.

Training, evaluation, inference, and Gradio modules are placeholders. There are no working application commands yet. Configuration paths are intended to resolve from the repository root. YAML settings are starting configurations, not validated hyperparameters. Derive the class count from the audited mapping before building models. Importing the ResNet18 module does not load or download weights.

## ResNet18 model and data contract

Create the model from the repository root, with the actual number of selected identities:

```python
from src.models.resnet18 import build_resnet18

model = build_resnet18(
    num_classes=50,  # Example only: replace with the audited identity count K.
    pretrained=True,
    trainable_backbone="none",
)
# logits = model(images)  # [B, K]; pass directly to torch.nn.CrossEntropyLoss.
```

`pretrained=True` requests `ResNet18_Weights.DEFAULT` from torchvision. The first construction may download weights if they are not cached; loading errors propagate without a random-weight fallback. These weights were pretrained on ImageNet, not on our selected identities. The replacement head is randomly initialized and adaptation to our dataset remains to be done. Use `pretrained=False` for offline implementation checks. No weights are bundled or downloaded as part of this change.

The returned module wraps the torchvision model as `model.network`; the replacement head is `model.network.fc`. Outputs are raw logits, without softmax. Preprocessing is not performed inside the model.

| Backbone mode | Trainable parameters | Behavior after `model.train()` |
|---|---|---|
| `"none"` | Classification head (`fc`) only | Entire backbone stays in eval mode; its BatchNorm statistics stay fixed |
| `"layer4"` | Last residual block (`layer4`) and head | Earlier blocks stay in eval mode; layer4 BatchNorm statistics can update |
| `"all"` | Entire model | All blocks enter training mode |

Call `model.train()` and `model.eval()` on the returned wrapper. No extra freeze helper is needed after `train()`. `eval()` puts all blocks in evaluation mode. Freezing parameters and freezing BatchNorm running statistics are handled together. To start a later fine-tuning stage while retaining learned weights, call `model.set_trainable_backbone("layer4")` and rebuild the optimizer with the desired parameter groups and learning rates. Do not call `model.network.train()` directly or manually change `requires_grad` to bypass the policy. State-dict keys have a `network.` prefix; reconstruct the wrapper with the saved configuration before loading a checkpoint.

Expected DataLoader batches:

- Floating-point images `[B, 3, H, W]`, with RGB channels; planned size is **128 x 128**.
- Convert pixel values to `[0, 1]`, then normalize **once in the data pipeline** with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- Labels of dtype `torch.long`, shape `[B]`, with integer values from `0` to `K - 1`; use the same contiguous identity mapping in every split.
- Training augmentation belongs only to training data. Validation and test preprocessing must be deterministic and consistent with the checkpoint settings.

`configs/resnet18.yaml` preserves the existing schema. Its `model.num_classes`, `model.pretrained`, and `model.trainable_backbone` correspond to builder arguments, but no configuration loader connects them automatically yet. `pretrained_weights: DEFAULT` documents the builder's fixed weight choice, and `replace_classifier_head: true` documents its always-on head replacement; neither is an extra builder argument. `num_classes: null` is intentionally unresolved and must be replaced by K before construction.

The remaining fields describe future integration: seed 42, batch size 32, AdamW, and a 15-epoch head-only baseline at learning rate `1e-3`. The fine-tuning section describes a separate later experiment at backbone learning rate `1e-4`, with no epoch budget selected yet. These are starting choices, not optimized settings.

With an existing Python environment containing torch and torchvision, run the offline checks from the repository root:

```powershell
python -m unittest tests.test_resnet18
```

These checks use synthetic RGB batches and `pretrained=False` to cover output shape, the three trainable-parameter policies, cross-entropy/backpropagation, frozen gradients, BatchNorm buffers, policy changes, and invalid arguments. Mocked weight-loading checks verify argument routing and failure propagation without a download. They validate the implementation, not face-recognition accuracy or actual pretrained-weight loading. Execution requires a working Python/PyTorch/torchvision environment; merely adding these tests does not constitute a passing run.

## Implementation roadmap

1. Audit identity labels and implement reproducible preprocessing/splitting in `src/data/prepare.py`, including duplicate/session grouping and leakage checks.
2. Implement datasets/loaders and training/evaluation transforms; verify samples and the shared class mapping.
3. Integrate the implemented ResNet18 into the shared training/validation workflow and implement the custom CNN. Tune on validation data and retain the best validation checkpoint.
4. Evaluate selected models on held-out test data and populate the analysis notebook and figures with real measurements.
5. Implement checkpoint-based image prediction and a Gradio image-upload interface that reuses inference code.
6. If time permits, separately scope and evaluate face detection integration, unknown rejection, and video tracking.

Future checkpoints must include model configuration, identity-to-class mapping, and preprocessing settings (input size, color mode, normalization, and crop/alignment policy). Also save weights, seed, training configuration, and dataset/split provenance. Inference must restore these settings and the original class order rather than guess them.
