# py-irt package

This code is an adaptation of the code in https://github.com/nd-ball/py-irt. 

## Installation

py-irt is now available on PyPi!

### Pre-reqs


Install [Poetry](https://python-poetry.org/docs/#installation)

```shell
git clone https://github.com/linggm3/py-irt.git
cd py-irt
poetry install
```



## Citations

If you use this code, please consider citing the following papers:

```shell
@inproceedings{lalor2019emnlp,
  author    = {Lalor, John P and Wu, Hao and Yu, Hong},
  title     = {Learning Latent Parameters without Human Response Patterns: Item Response Theory with Artificial Crowds},
  year      = {2019},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
}
```

```shell
@inproceedings{rodriguez2021evaluation,
  title={Evaluation Examples Are Not Equally Informative: How Should That Change NLP Leaderboards?},
  author={Rodriguez, Pedro and Barrow, Joe and Hoyle, Alexander Miserlis and Lalor, John P and Jia, Robin and Boyd-Graber, Jordan},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  pages={4486--4503},
  year={2021}
}
```

Implementation is based on the following paper:

```shell
@article{natesan2016bayesian,
  title={Bayesian prior choice in IRT estimation using MCMC and variational Bayes},
  author={Natesan, Prathiba and Nandakumar, Ratna and Minka, Tom and Rubright, Jonathan D},
  journal={Frontiers in psychology},
  volume={7},
  pages={1422},
  year={2016},
  publisher={Frontiers}
}
```

Email: john.lalor@nd.edu
