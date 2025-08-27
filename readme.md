<h1 align="center">CanonSwap</h1>

## Environment Setup

```bash
conda create -n CanonSwap python=3.10
conda activate CanonSwap
```

**Install PyTorch (Over versions may be supported):**

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

**Install other dependencies:**

```bash
pip install -r requirements.txt
```

## Model Download

### 1. CanonSwap Checkpoints
Download from [here](https://drive.google.com/file/d/1uDWiIam1jziU918iOZY2ATE2dw9aqYAr/view?usp=drive_link) and move to `./pretrained_weights` folder

### 2. InsightFace Models
Download Antelope from [here](https://drive.google.com/file/d/1yXQs6Nd0_hp97UGCvceyGehlCVTPs1ZD/view?usp=sharing).

Download buffalo_l from [here](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip).

Extract both models to `./pretrained_weights/insightface/models` with the following structure:
```
/pretrained_weights/insightface/models/
├── antelope/
│   ├── glintr100.onnx
│   └── scrfd_10g_bnkps.onnx
└── buffalo_l/
│   ├── 2d106det.onnx
│   └── det_10g.onnx
```


### 3. ArcFace
Download ArcFace from [here](https://drive.google.com/file/d/1lDpbmvc7__cIfWU9rTTKNW5OXeeqohUJ/view?usp=drive_link) and extract to `./pretrained_weights` folder.

### 4. Landmark Model
Download Landmark Model from [here](https://drive.google.com/file/d/1uuee7ebWr9lBYfCmIPk8c4fk_YDbiNHz/view?usp=drive_link) and extract to `./pretrained_weights` folder.


## Project Structure After Download

After downloading all models, your project structure should look like:

```
CanonSwap/
├── pretrained_weights/
│   ├── combined_weights.pth
│   ├── arcface_checkpoint.tar
│   ├── landmark.onnx
│   └── insightface/
│       └── models/
│           ├── antelope/
│           │   ├── glintr100.onnx
│           │   └── scrfd_10g_bnkps.onnx
│           └── buffalo_l/
│               ├── 2d106det.onnx
│               └── det_10g.onnx
```

## Inference
The first inference run will automatically download the face parsing model.
### Face Swapping
```bash
python inference_canswap.py -s examples/source.jpeg -t examples/target.mp4
```
This also supports image-to-image swapping.

### Video-to-Image Swap
```bash
python inference_v2i.py -s examples/i2v_s.jpeg -t examples/i2v_t.mov
```

## Citation

If you find this work useful, please star and cite:

```bibtex
@article{luo2025canonswap,
   title={CanonSwap: High-Fidelity and Consistent Video Face Swapping via Canonical Space Modulation},
   author={Luo, Xiangyang and Zhu, Ye and Liu, Yunfei and Lin, Lijian and Wan, Cong and Cai, Zijian and Huang, Shao-Lun and Li, Yu},
   journal={arXiv preprint arXiv:2507.02691},
   year={2025}
}
```
It is the greatest appreciation of our work!

## License and Attention
This project is licensed under the Research Responsible AI License (ResearchRAIL-M). A full copy of the license can be found in the LICENSE file.
By using this code or model, you agree to the terms outlined in the license.
This project is intended for technical and academic purposes only. You are strictly prohibited from using this project for any illegal or unethical applications, including but not limited to creating non-consensual content, spreading misinformation, or harassing individuals. Please see the Use-Based Restrictions in the license for a non-exhaustive list of prohibited uses.
The authors and copyright holders are exempt from any liability arising from your violation of these terms.

## Acknowledgments

This project is based on [LivePortrait](https://github.com/KwaiVGI/LivePortrait). We thank the authors for their excellent work in efficient portrait animation.
