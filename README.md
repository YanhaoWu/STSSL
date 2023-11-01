*-----------------------------------------------------------------------------------*


Update 19:56 8.21 2023 


Code is released, modify the 'epochs' to 'stop_epoch' in trainer.stssl_trainer 


*-----------------------------------------------------------------------------------* 


**(STSSL) Spatiotemporal Self-supervised Learning for Point Clouds in the Wild **

**[Paper](https://arxiv.org/pdf/2303.16235.pdf)** **|** **[Project page](https://yanhaowu.github.io/STSSL/)**

![](pics/poster.png)


Our project is built based on **[SegContrast](https://github.com/PRBonn/segcontrast)**

Installing pre-requisites:

`sudo apt install build-essential python3-dev libopenblas-dev`

`pip3 install -r requirements.txt`

`pip3 install torch ninja`

Installing MinkowskiEngine with CUDA support:

`pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps`


# Data Preparation

Download [KITTI](http://www.semantic-kitti.org/dataset.html#download) inside the directory ```your config.dataset_path/datasets```. The directory structure should be:

```
 ── your config.dataset_path/
    └── dataset
        └── sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
            └── ...
```


# Reproducing the results

for pre-training

you can just run train_stssl.py which is in tools, remember to modify the paramters of path : ) 

Then for fine-tuning:

you can just run train_downstream.py which is in tools, remember to adjust the learning rate: ) 

Of course, you can also refer to **[SegContrast](https://github.com/PRBonn/segcontrast)**

Any questions, touch me at wuyanhao@stu.xjtu.edu.cn


# Citation

If you use this repo, please cite as :

```
@inproceedings{wu2023spatiotemporal,
  title={Spatiotemporal Self-supervised Learning for Point Clouds in the Wild},
  author={Wu, Yanhao and Zhang, Tong and Ke, Wei and S{\"u}sstrunk, Sabine and Salzmann, Mathieu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5251--5260},
  year={2023}
}
```
