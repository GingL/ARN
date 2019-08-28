# PyTorch Implementation of ARN

## Introduction

This repository is Pytorch implementation of [Adaptive Reconstruction Network for Weakly Supervised Referring Expression Grounding]() in ICCV 2019.
Check our [paper]() for more details.

## Prerequisites

* Python 3.5
* Pytorch 0.4.1
* CUDA 8.0

## Installation

Please refer to [MattNet](https://github.com/lichengunc/MAttNet) to install [mask-faster-rcnn](https://github.com/lichengunc/mask-faster-rcnn), [REFER](https://github.com/lichengunc/refer) and [refer-parser2](https://github.com/lichengunc/refer-parser2).
Follow Step 1 & 2 in Training to prepate the data and features.

## Training

Train ARN with ground-truth annotation:

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/train.py --dataset ${DATASET} --splitBy ${SPLITBY} --exp_id ${EXP_ID}
```

## Evaluation

Evaluate ARN with ground-truth annotation:

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval.py --dataset ${DATASET} --splitBy ${SPLITBY} --split ${SPLIT} --id ${EXP_ID}
```


## Citation

    @inproceedings{lxj2019arn,
      title={Adaptive Reconstruction Network for Weakly Supervised Referring Expression Grounding},
      author={Xuejing Liu, Liang Li, Shuhui Wang, Zheng-Jun Zha, Dechao Meng, and Qingming Huang},
      booktitle={ICCV},
      year={2019}
    }


## Acknowledgement

Thanks for the work of [Licheng Yu](http://cs.unc.edu/~licheng/). Our code is based on the implementation of [MattNet](https://github.com/lichengunc/MAttNet).

## Authorship

This project is maintained by [Xuejing Liu](https://gingl.github.io/).
