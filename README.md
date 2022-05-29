# TrueType Transformer : ***T <sup> 3 </sup>***

This is an official PyTorch implementation of the paper TrueType Transformer: Character and Font Style Recognition in Outline Format, which is accepted to **DAS2022**.

***T <sup> 3 </sup>*** is a recognition model, input as an outline format (i.e. True type format) and output as a class label.

# Installation
```bash
pip install -r /requirements.txt
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
```
### A file tree
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

In experiments, we used [Googlefonts](https://github.com/google/fonts.git).
Please be cautious that we modified these datasets in the way mentioned in the paper.

# Note

注意点などがあれば書く

# Author

作成情報を列挙する

* 作成者
* 所属
* E-mail

# License
ライセンスを明示する

"hoge" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

社内向けなら社外秘であることを明示してる

"hoge" is Confidential.