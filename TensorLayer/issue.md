[Shall we add __all__ attribute in __init__.py · Issue #405 · tensorlayer/tensorlayer][5]

### Some reasons:

* A quote from "D. Beazley, B.K. Jones - Python Cookbook, 3rd Edition. 2013"
> Although the use of from module import * is strongly discouraged, it still sees frequent use in modules that define a large number of names. If you don’t do anything, this form of import will export all names that don’t start with an underscore. On the other hand, if __all__ is defined, then only the names explicitly listed will be exported. If you define __all__ as an empty list, then nothing will be exported. An AttributeError is raised on import if __all__ contains undefined names.

* [syntax - Can someone explain __all__ in Python? - Stack Overflow](https://stackoverflow.com/questions/44834/can-someone-explain-all-in-python)
* Which names will be exported? 
```python
from tensorlayer.layers import *
```
* `_load_mnist_dataset(shape, path)` should not be valid

![test example][4]

### Some example:
* [`pytorch/__init__.py` at master · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/master/torch/__init__.py)

![PyTorch][1]

* [`pytorch/__init__.py` at master · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/__init__.py)

![PyTorch][2]

* [`scikit-learn/__init__.py` at master · scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/__init__.py)

![scikit-learn][3]


[1]: http://ozesj315m.bkt.clouddn.com/img/q1.png
[2]: http://ozesj315m.bkt.clouddn.com/img/p2.png
[3]: http://ozesj315m.bkt.clouddn.com/img/q12.png
[4]: http://ozesj315m.bkt.clouddn.com/img/pa.png
[5]: https://github.com/tensorlayer/tensorlayer/issues/405
