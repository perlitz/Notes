# Python crash course

IPython has a magic function %paste which correctly pasts.

start a new conda environment:

```bash
conda create -n ~VenvName~ python=3.6
```

Moving into one comes down to 

```bash
conda activate ~VenvName~
```



 Using **defaultdict** is a handy way to use a dictionary that has a default value and will not raise an error when called with a nonexistent key. `defaultdict(int)` will return a 0 whenever called with an empty key since `int()=0`

**Sorting** - there is a difference between the `y=sorted(x)` and `y = x.sort()`, in the first case, the sorting will not change x, in the second, it will. Sorting can be done with all kinds of keys, some defined such as `key=abs` and some can be defined by the user using a designated predefined function.

##### Assert statements

An assertion looks like:

```python
assert 1+1 == 2, "1+1 should equal 2 but doesn't"
```

**Use asserts to check if our functions do what we expect them to do**

##### Args and Kwargs

When we want to define a function that takes an arbitrary number of arguments, we can define it as:

```python
def magic(*args, **kwargs):
	print(f"unnamed args: {args}")
    print(f"keyword args: {kwargs}")
    
magic(1, 2. key="word", key2="word2")
```

This option is a kind of last resort, it is better to explicitly define the arguments whenever possible.

##### Type annotations

Some types of data structures can be used using the default types, however, if we want to use some other types, we can import this type from the `typing` library.

```python
from typing import List

def total(xs: List[float]) -> float:
    return sum(total)
```

Other types are `Callable` for functions, `Tuple`, `Dict`.