# Patch-based 3D Natural Scene Generation from a Single Example (CVPR 2023)

:star: Generating (and Editing) diverse 3D natural scenes from a single example **without** any training.

#####  <p align="center"> [Weiyu Li*](https://wyysf-98.github.io/), [Xuelin Chen*†](https://xuelin-chen.github.io/), [Jue Wang](https://juewang725.github.io/), [Baoquan Chen](https://cfcs.pku.edu.cn/baoquan/)</p>
 
#### <p align="center">[Project Page](https://wyysf-98.github.io/Sin3DGen) | [ArXiv](https://arxiv.org/abs/2304.12670) | [Paper](https://wyysf-98.github.io/Sin3DGen/paper/Paper_high_res.pdf) | [Supp_material](https://wyysf-98.github.io/Sin3DGen/paper/Supplementary_high_res.pdf) | [Video](https://youtu.be/qahByVuhLJw)</p>

<p align="center">
  <a href="https://colab.research.google.com/github/wyysf-98/Sin3DGen/blob/main/colab_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg">
  </a>
</p>

<p align="center">
  <img src="https://wyysf-98.github.io/Sin3DGen/assets/images/teaser_bak.png"/>
  <p align="center">High-quality 3D scenes created by our method (background sky post-added)</p>
</p>


## Prerequisite

<details> <summary>Setup environment</summary>

:smiley: We also provide a Dockerfile for easy installation, see [Setup using Docker](./docker/README.md).

 - Python 3.8
 - PyTorch 1.9.1
 - [svox2](https://github.com/sxyu/svox2)
 - [unfoldNd](https://github.com/f-dangel/unfoldNd)
 - [Open3D](https://github.com/isl-org/Open3D) for visualization

Clone this repository.

```sh
git clone git@github.com:wyysf-98/Sin3DGen.git
```

Install the required packages.

```sh
conda create -n Sin3DGen python=3.8
conda activate Sin3DGen
conda install -c pytorch pytorch=1.9.1 torchvision=0.10.1 cudatoolkit=10.2 && \
conda install -c bottler nvidiacub && \
pip install -r docker/requirements.txt
```


</details>

<details> <summary>Data preparation</summary>

We provide some Plenoxels scenes and optimized mapping fields in [link]() for a quick test. Please download and unzip to current folder. Then the folder should as following:

```
└── data
    └── DevilsTower
        ├── mapping_fields
        |   ├── ...
        |   └── sxxxxxx.npz     # Synthesized mapping fields
        └── ckpts
            ├── rgb_fps8.mp4    # Visualization of the scene
            ├── ckpt_reso.npz   # Plenoxels saving files
            └── mesh_reso.obj   # Extracted meshes
```

</details>

<details> <summary>Use your own data*</summary>

Please refer to [svox2](https://github.com/sxyu/svox2) to prepare your own data. 
You can also use blender to render scenes as in [NSVF](https://github.com/facebookresearch/NSVF/blob/main/examples/data/nerf_render_ori.py).

\* Note that all scenes must be inside `a unit box centered at the origin`, as mentioned in the paper.

Then you should get your scenes using our forked version [Link](https://github.com/wyysf-98/svox2).

The main differences of the original version are:
  - We made modifications to certain parts of [opt.py](https://github.com/sxyu/svox2/blob/master/opt/opt.py) to enable the preservation of intermediate checkpoint during the training process.
  - Add more stages during training in configuration.

```sh
git clone git@github.com:wyysf-98/svox2.git
cd svox2
./launch.sh {yout_data_name} 0 {yout_data_path} -c configs/syn_start_from_12.json
```

</details>

## Quick inference demo
For local quick inference demo using optimized mapping field, you can use

```sh
python quick_inference_demo.py -m 'run' \
      --config './configs/default.yaml' \
      --exemplar './data/DevilsTower/ckpts' \
      --resume './data/DevilsTower/mapping_fields/s566239.npz' \
      --output './outputs/quick_inference_demo/DevilsTower_s566239' \
      --scene_reso '[512, 512, 512]' # resolution for visualization, change to '[384, 384, 384]' or lower when OOM
```

## Optimization
We provide a colab for a demo
<p>
  <a href="https://colab.research.google.com/github/wyysf-98/Sin3DGen/blob/main/colab_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg">
  </a>
</p>

We use a NVIDIA Tesla V100 with 32 GB Ram to generate the novel scenes, which takes about 10 mins as mentioned in our paper.

```sh
python generate.py -m 'run' \
      --config './configs/default.yaml' \
      --exemplar './data/DevilsTower/ckpts' \
```

if you encounter OOM problem, try to reduce `pyr_reso` for synthesis by adding `--pyr_reso [ 16, 21, 28, 38, 51, 68, 91]` or the `scene_reso` for visualization by adding `--scene_reso [216, 216, 216]`.
For more configurations, please refer to the comments in the `configs/default.yaml`.


## Evaluation
We provide the relevant code for evaluating the metrics (SIFID, SIMMD, image_diversity, scene_diversity), please change the evaluation script based on your actual situation.
```sh
cd evaluation
python compute_metrics.py --exp {out_path} \
                          --img_gt {GT_images_path} \
                          --mesh_gt {GT_mesh_path} \
                          --out_dir ./results/{exp_name}
```

## Acknowledgement

The implementation of `exact_search.py` and evaluation for images partly took reference from [Efficient-GPNN](https://github.com/ariel415el/Efficient-GPNN). We thank the authors for their generosity to release code.


## Citation

If you find our work useful for your research, please consider citing using the following BibTeX entry.

```BibTeX
@article{weiyu23Sin3DGen,
    author    = {Weiyu Li and Xuelin Chen and Jue Wang and Baoquan Chen},
    title     = {Patch-based 3D Natural Scene Generation from a Single Example},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023},
}
```