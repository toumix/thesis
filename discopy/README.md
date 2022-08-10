# DisCoPy: Python for the applied category theorist

DisCoPy (Distributional Compositional Python) is a toolbox for applied category theory.

## Test

```shell
pip install -r test/requirements.txt
coverage run -m pytest --doctest-modules --pdb
coverage report -m discopy/*.py
```
