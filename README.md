# TrueType Transformer : ***T <sup> 3 </sup>***

This is an official PyTorch implementation of the paper TrueType Transformer: Character and Font Style Recognition in Outline Format, which is accepted to **DAS2022**.

***T <sup> 3 </sup>*** is a recognition model, input as an outline format (i.e. True type format) and output as a class label.

# Installation
```bash
pip install -r ./requirements.txt
```
# Usage
## Recommendation
- Git 2.25.1
- Docker 20.10.13
## A sample of to create a development environment
```bash
mkdir -p ~/Dev/data && cd ~/Dev/data
git clone https://github.com/google/fonts.git
cd ../
git clone ${This repository}
cd ./T3
sh docker/build.sh
sh docker/run.sh
sh docker/exec.sh
sh src/run.sh
```
### File tree
```bash
.
├── T3
│   ├── docker
│   ├── models
│   ├── src
│   │   ├── conf
│   │   │   └── config.yaml
│   │   ├── model
│   │   │   └── T3.py
│   │   ├── utils
│   │   │   ├── evaluate.py
│   │   │   ├── load.py
│   │   │   └── train.py
│   │   ├ run.sh
│   │   └ main.py
│   ├── .gitignore
│   ├── .env
│   ├── README.md
│   ├── Dockerfile
│   ├── DockerfileM1mac
│   ├── google_font_category_v4.csv
│   ├── requirements.txt
│   └── estimate.ipynb
└── data
    └── fonts
```
## Dataset

In experiments, we used [Googlefonts](https://github.com/google/fonts.git).\
Please be cautious that we modified these datasets followed [STEFANN](https://prasunroy.github.io/stefann/) for data split.

## Execution scripts
### Models
* Proposed: refer to `T3.py`. You can change hyper parameter in the paper by `config.yaml`.
* Estimate: refer to `estimate.ipynb`.

For a training example, `src/run.sh`.


# Author
* Yusuke Nagata, Jinki Otao, Daichi Haraguchi and Seiichi Uchida
* E-mail
  * yusuke.nagata@human.ait.kyushu-u.ac.jp
  * jinki.otao@human.ait.kyushu-u.ac.jp
  * daichi.haraguchi@human.ait.kyushu-u.ac.jp
  * uchida@ait.kyushu-u.ac.jp

# Citation
[Y. Nagata, J. Otao, D. Haraguchi and S. Uchida "TrueType Transformer: Character and Font Style Recognition in Outline Format." Document Analysis Systems: 15th IAPR International Workshop, DAS 2022, La Rochelle, France, May 22–25, 2022, Proceedings. 2022.](https://link.springer.com/chapter/10.1007/978-3-031-06555-2_2)

```bash
@inproceedings{nagata2022truetype,
  title={TrueType Transformer: Character and Font Style Recognition in Outline Format},
  author={Yusuke Nagata and Jinki Otao and Daichi Haraguchi and Seiichi Uchida},
  booktitle={International Workshop on Document Analysis Systems},
  pages={18--32},
  year={2022},
  organization={Springer}
}
```
# License
Follow google fonts for the license of the dataset.