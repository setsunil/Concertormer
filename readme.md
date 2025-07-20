# Concertormer

The official pytorch implementation of the papers  
**[Efficient Concertormer for Image Deblurring and Beyond (ICCV 2025)](https://arxiv.org/abs/2404.06135)**

#### [**Pin-Hung Kuo**](https://setsunil.github.io/), [Jinshan Pan](https://jspan.github.io/), [Shao-Yi Chien](https://www.ee.ntu.edu.tw/profile1.php?teacher_id=943013&p=3), and [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)


### Datasets
Please follow the instructions of [Restormer](https://github.com/swz30/Restormer) and [NAFNet](https://github.com/megvii-research/NAFNet).

### Usage
* Train
```
python -m torch.distributed.run --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/<dataset>/<model>.yml --launcher pytorch
```

* Test
```
python -m torch.distributed.run --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/<dataset>/<model>.yml --launcher pytorch
```

* Evaluation
For RealBlur, please use the python script evaluate_realblur.py
```
python evaluate_realblur.py --folder RealBlurR --dataset R
python evaluate_realblur.py --folder RealBlurJ --dataset J
```

For deraining, please use the m-file evaluate_derain.m
```
matlab -nodesktop -nosplash -r "evaluate_derain('Derain')"
```


### Pretrained Models
[Google Drive](https://drive.google.com/file/d/1NvbGroZm4vVgvWJtmgW-6Fwd97PiukiL/view?usp=drive_link)

### Results
You can also download our results directly.  
[Google Drive](https://drive.google.com/file/d/1iqLNR6yrHnL2g2nux5zHKb90gnuTBHOq/view?usp=drive_link)

### Citations
If this paper helps your research or work, please consider citing us.

```
@article{kuo2024efficient,
  title={Efficient Concertormer for Image Deblurring and Beyond},
  author={Kuo, Pin-Hung and Pan, Jinshan and Chien, Shao-Yi and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:2404.06135},
  year={2024}
}
```

### Contact

If you have any questions, please contact setsunil@gmail.com

---
<!--
<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.glitch.me/badge?page_id=megvii-research/NAFNet)

</details>

-->