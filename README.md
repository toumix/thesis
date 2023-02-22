# Category Theory for Quantum Natural Language Processing

This repository contains the LaTeX sources for my PhD thesis (which is now available on [arXiv:2212.06615](https://arxiv.org/abs/2212.06615)) together with a standalone version of [DisCoPy](https://github.com/discopy/discopy) intended to make the thesis self-contained.

## Test

```shell
pip install -r test/requirements.txt
coverage run -m pytest --doctest-modules --pdb
coverage report -m discopy/*.py
```
