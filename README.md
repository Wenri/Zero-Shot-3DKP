# Zero-Shot-3DKP
## ZeroKey: Point-Level Reasoning and Zero-Shot 3D Keypoint Detection from Large Language Models

### [Bingchen Gong](https://s2.hk/)<sup>1</sup>, [Diego Gomez](https://www.lix.polytechnique.fr/~gomez/)<sup>1</sup>, [Abdullah Hamdi](https://abdullahamdi.com/)<sup>2</sup>, [Abdelrahman Eldesokey](https://abdo-eldesokey.github.io/)<sup>3</sup>, [Ahmed Abdelreheem](https://samir55.github.io/)<sup>3</sup>, [Peter Wonka](https://peterwonka.net/)<sup>3</sup>, [Maks Ovsjanikov](https://www.lix.polytechnique.fr/~maks/)<sup>1</sup>
<sup>1</sup>École Polytechnique, <sup>2</sup>University of Oxford, <sup>3</sup>KAUST

[**Project Website**](https://sites.google.com/view/zerokey) | [PDF](https://arxiv.org/pdf/2412.06292)


---

**Abstract:**
We propose a novel zero-shot approach for keypoint detection on 3D shapes. Point-level reasoning on visual data is challenging as it requires precise localization capability, posing problems even for powerful models like DINO or CLIP. Traditional methods for 3D keypoint detection rely heavily on annotated 3D datasets and extensive supervised training, limiting their scalability and applicability to new categories or domains. In contrast, our method utilizes the rich knowledge embedded within Multi-Modal Large Language Models (MLLMs). Specifically, we demonstrate, for the first time, that pixel-level annotations used to train recent MLLMs can be exploited for both extracting and naming salient keypoints on 3D models without any ground truth labels or supervision. Experimental evaluations demonstrate that our approach achieves competitive performance on standard benchmarks compared to supervised methods, despite not requiring any 3D keypoint annotations during training. Our results highlight the potential of integrating language models for localized 3D shape understanding. This work opens new avenues for cross-modal learning and underscores the effectiveness of MLLMs in contributing to 3D computer vision challenges.

![Keypoint Detection using BT3D](https://wimmerth.github.io/b2-3d/static/images/qualitative_results_5.png)

---

**Installation:**

```bash
conda create -n zerokey python=3.12
conda activate zerokey
conda install pytorch torchvision pytorch-cuda xformers -c pytorch -c nvidia -c xformers
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip install scipy scikit-learn pandas potpourri3d
```
Setting up the environment is a bit tricky, as always with Python packages.
We found that the above setup works well for us, but you might need to adjust the versions of the packages to match your
system (CUDA etc.).
The installation notes for [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) and for
[DINOv2](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#installation) might be helpful for this.
If you'd like to use the [SAM](https://github.com/facebookresearch/segment-anything) or the
[CLIP](https://github.com/openai/CLIP) model as feature extractors, follow the instructions in the respective
repositories to install the additionally required packages.

**Experiment:**

To run the experiments on the KeypointNet dataset, you need to
first [download the dataset](https://github.com/qq456cvb/KeypointNet).
Next, set the environment variable `export KEYPOINTNET_DATASET_PATH="</path/to/KeypointNet/dataset>"`.
You should now be able to run the evaluation using `python3 kpviews.py`.

---

**Citation:**

```bibtex
@misc{gong2024zerokeypointlevelreasoningzeroshot,
      title={ZeroKey: Point-Level Reasoning and Zero-Shot 3D Keypoint Detection from Large Language Models},
      author={Bingchen Gong and Diego Gomez and Abdullah Hamdi and Abdelrahman Eldesokey and Ahmed Abdelreheem and Peter Wonka and Maks Ovsjanikov},
      year={2024},
      eprint={2412.06292},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.06292},
}
```
