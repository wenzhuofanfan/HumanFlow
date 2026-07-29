# HumanFlow

<p align="center">
  <strong>HumanFlow: Controllable Human Image Generation via Flow Matching</strong>
</p>

<p align="center">
  Wenzhuo Fan · Hongsheng Zheng · Jianchi Sun · Fei Fang · Hong Ding · Chunxia Xiao
</p>

<p align="center">
  <a href="https://doi.org/10.1145/3811361">
    <img src="https://img.shields.io/badge/Paper-ACM%20TOG-blue" alt="Paper">
  </a>
  <img src="https://img.shields.io/badge/Code-Coming%20Soon-orange" alt="Code Coming Soon">
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/Backbone-FLUX.1--dev-purple" alt="FLUX.1-dev">
</p>

## 🚧 Release Status

> [!IMPORTANT]
> The official implementation of **HumanFlow** is currently being organized and will be released in this repository.
>
> We plan to progressively release the inference code, pretrained checkpoints, training scripts, control-condition preprocessing tools, evaluation code, and documentation.
>
> Please **star** or **watch** this repository to receive future updates.

## Overview

**HumanFlow** is a unified flow-matching-based framework for high-fidelity and controllable full-body human image generation.

Given a textual description and a human-centric control condition, HumanFlow generates realistic full-body human images while preserving both:

- Semantic consistency with the text prompt;
- Structural alignment with the input control condition;
- Anatomically plausible human body topology;
- High visual fidelity under challenging poses and occlusions.

HumanFlow supports multiple human-centric control modalities within a unified framework:

- Human keypoints;
- Semantic segmentation;
- Canny edges;
- Depth maps;
- DensePose;
- Surface normal maps.

<!--
Add the teaser image after uploading it to the repository:

<p align="center">
  <img src="assets/teaser.jpg" width="95%">
</p>
-->


### MiCoGen Dataset

We construct **MiCoGen**, a large-scale multi-condition dataset for controllable human image generation.

MiCoGen contains **1,019,124 full-body human images**. Each image is paired with:

- A textual description;
- Human keypoints;
- Semantic segmentation;
- Canny edges;
- Depth maps;
- DensePose;
- Surface normal maps.

All control annotations are spatially aligned with their corresponding images.

The dataset construction pipeline includes:

1. Data collection from public resources, existing datasets, and captured images;
2. Filtering based on resolution, image quality, viewpoint, body completeness, and background complexity;
3. Text-description generation;
4. Multi-modal control-condition generation;
5. Iterative manual inspection and annotation-pipeline optimization.

<!--
Add the dataset pipeline after uploading the figure:

<p align="center">
  <img src="assets/micogen_pipeline.jpg" width="95%">
</p>
-->
Download：https://huggingface.co/datasets/fwz0818/MiCoGen


## Experimental Configuration

The main experiments use the following configuration:

| Setting | Value |
|---|---|
| Backbone | FLUX.1-dev |
| Image resolution | 512 × 512 |
| Framework | PyTorch |
| Precision | bfloat16 |
| Optimizer | AdamW |
| Batch size | 16 |
| Training steps | 100,000 |
| Topology templates | 5 |
| Keypoint format | COCO 17 keypoints |
| Training hardware | NVIDIA RTX PRO 6000 96 GB |
| Training samples | 1,000,000 |
| Evaluation samples | 19,124 |

The reported training time is approximately 45 hours for each control condition under the paper's experimental environment.

## Installation

Installation instructions will be provided with the official code release.

```bash
# Coming soon
```

The released environment is planned to include dependencies for:

- PyTorch;
- Diffusers;
- Transformers;
- FLUX.1-dev;
- Accelerate;
- OpenCV;
- Keypoint estimation;
- Segmentation and DensePose processing.

## Pretrained Models

Pretrained checkpoints will be released progressively.

| Control condition | Checkpoint |
|---|---|
| Keypoints | Coming soon |
| Segmentation | Coming soon |
| Canny | Coming soon |
| Depth | Coming soon |
| DensePose | Coming soon |
| Surface normals | Coming soon |

## Inference

Inference scripts and example configurations will be uploaded soon.

The planned inference interface will accept a text prompt and one human-centric control condition:

```text
Text prompt + Control image → Generated full-body human image
```

Example:

```text
Prompt:
A woman is wearing a white sweater with a plaid skirt,
high boots, and a beret-style hat.

Control:
assets/examples/keypoints.png
```

Planned command-line usage:

```bash
python inference.py \
    --prompt "A woman is wearing a white sweater with a plaid skirt, high boots, and a beret-style hat." \
    --control_type keypoints \
    --control_image assets/examples/keypoints.png \
    --output outputs/result.png
```

> The command above is a preview of the planned interface and may change in the official release.

## Training

The training release is planned to include:

- MiCoGen data-format documentation;
- Control-condition preprocessing;
- FLUX backbone initialization;
- Control Encoder implementation;
- Token-ControlNet implementation;
- HTCL implementation;
- Distributed and mixed-precision training;
- Checkpoint saving and resuming;
- Validation and evaluation scripts.

Planned usage:

```bash
accelerate launch train.py \
    --config configs/humanflow_keypoints.yaml
```

> The final configuration names and command-line arguments will be documented after the code is released.

## Repository Roadmap

- [ ] Release environment and dependency files
- [ ] Release inference code
- [ ] Release pretrained checkpoints
- [ ] Release example control conditions
- [ ] Release Control Encoder and Token-ControlNet
- [ ] Release HTCL implementation
- [ ] Release training code
- [ ] Release evaluation scripts
- [ ] Release MiCoGen data-processing instructions
- [ ] Add an interactive demo
- [ ] Add additional control modalities

## Limitations

HumanFlow currently has several limitations:

- MiCoGen relies partly on automatically generated control annotations, and inaccurate annotations may introduce localized inconsistencies;
- Fine-grained appearance details such as clothing textures, accessories, and small patterns remain challenging;
- Missing or incomplete regions in geometric control maps may lead to incorrect generation;
- Although the architecture is modality-agnostic, new control modalities may require specialized representations or conditioning designs;
- Current experiments primarily focus on single full-body human image generation.

## Citation

Please cite our work if you find this project useful:

```bibtex
@article{fan2026humanflow,
  title   = {HumanFlow: Controllable Human Image Generation via Flow Matching},
  author  = {Fan, Wenzhuo and Zheng, Hongsheng and Sun, Jianchi and
             Fang, Fei and Ding, Hong and Xiao, Chunxia},
  journal = {ACM Transactions on Graphics},
  volume  = {45},
  number  = {4},
  articleno = {1},
  pages   = {1--16},
  year    = {2026},
  month   = {July},
  doi     = {10.1145/3811361}
}
```

## Contact

For questions about the paper or the future code release, please contact:

- **Wenzhuo Fan:** fwz0818@whu.edu.cn
- **Chunxia Xiao:** cxxiao@whu.edu.cn

GitHub issues will also be welcomed after the official code release.

## License

The paper is distributed under the **Creative Commons Attribution 4.0 International License**.

The licenses for the source code, pretrained checkpoints, and dataset-related resources will be provided when the corresponding materials are released.

Users must also follow the licenses and usage conditions of all third-party components, including FLUX.1-dev and the datasets used in this project.

## Acknowledgements

HumanFlow builds upon research and open-source projects in:

- Flow matching;
- FLUX;
- Controllable image generation;
- Human pose estimation;
- Human segmentation;
- DensePose;
- Human-centric foundation models.

We sincerely thank the authors and maintainers of these projects for their valuable contributions.
