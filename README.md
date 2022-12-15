# Category Theory for Quantum Natural Language Processing

This repository contains the LaTeX sources for my PhD thesis together with a standalone version of [DisCoPy](https://github.com/discopy/discopy) intended to make the thesis self-contained.

## Test

```shell
pip install -r test/requirements.txt
coverage run -m pytest --doctest-modules --pdb
coverage report -m discopy/*.py
```
