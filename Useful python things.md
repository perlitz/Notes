# Useful python things

## Small things

#### plot a function using simpy

```python
from sympy import symbols
from sympy import plot

t = symbols('t')
x = 1/(1+t/2)

plot(x, (t, 0, 10), ylabel='Speed')
```



#### Define an object using namedtuple

```python
from collections import namedtuple
gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])
```



#### Underscoring and stuff

`__foo__`: this is just a convention, a way for the Python system to use names that won't conflict with user names.

`_foo`: this is just a convention, a way for the programmer to indicate that the variable is private (whatever that means in Python).

`__foo`: this has real meaning: the interpreter replaces this name with `_classname__foo` as a way to ensure that the name will not overlap with a similar name in another class.

No other form of underscores have meaning in the Python world.

There's no difference between class, variable, global, etc in these conventions.

#### np.random.seed()

The object np.random is a generator of random numbers, a generator in python is a distinct object in that one can run over it (like an regular iterator) with each object gets calculated on the fly. It is important to note in the case of random generator that once a seed is defined there is a deterministic chain of numbers outputted, if one wants the same output from a certain function that uses randomization given the same input, one has to define the seed only once.

```python
# Define a seed for numpy and tensorflow, this has to happen everytime one calls the random class.
np.random.seed(0)
tf.set_random_seed(0)
```



## Read all files in tree and convert them to another format

In this example, I convert from .npy format to .mat format but this can be easily changed to other formats

 ```python
import scipy.io, os, pickle
import numpy as np

input_dir_path = '/raid/algo/SOCISP_SLOW/ADAS/PD/Cityscapes/results/inference/SIRC_classical_PD/v0.24/bboxes'


output_root_dir_path = '/home/yotampe/Code/Checkerboards/Checkerboards_CVPR15_codebase/bboxs'

# read all files in a directory and append them to file_paths
li

# Iterate over all file paths, read the .npy file and write it as .mat
for input_file_path in input_file_paths:
    input_file_data = np.load(open(input_file_path, "rb"))
    output_dir_path = os.path.join(output_root_dir_path, 				 						input_file_path.split('/')[-3], input_file_path.split('/')[-2])

    if not os.path.isdir(output_dir_path):
        os.makedirs(output_dir_path)

    output_file_path = os.path.join(output_dir_path, 					 							input_file_path.split('/')[-1])
    output_file_path = output_file_path.split('.')[0]
    scipy.io.savemat(output_file_path, mdict={'data': input_file_data})
 ```



## Read a txt file, arrange something and write to another

```python
import os

in_txt_dir_path = '/home/yotampe/Code/Checkerboards/Checkerboards_CVPR15_codebase/output/OutTxt_v0.2'
out_txt_dir_path = '/home/yotampe/Code/Checkerboards/Checkerboards_CVPR15_codebase/output/OutTxt_v0.2_fixed_confidence'

os.mkdir(out_txt_dir_path)

txt_files = [file_name for file_name in os.listdir(in_txt_dir_path) if file_name.endswith('.txt')]

for txt_file_name in txt_files:


    with open(in_txt_dir_path + '/' + txt_file_name) as read_txt_file:
        with open(out_txt_dir_path + '/' + txt_file_name,'w+') as write_txt_file:

            line = read_txt_file.readline()
            while line:
     			# do something to line here
                 write_txt_file.write(line)
                 line = read_txt_file.readline()
```

## A more general option to iterate over files

```python
import os

class Convert(object):

    def __init__(self,in_dir_path,out_dir_path):

        self.in_dir_path = in_dir_path
        self.out_dir_path = out_dir_path

        assert os.path.isdir(self.in_dir_path),"Input dir not valid"

        if not os.path.isdir(self.out_dir_path):
            print("Creating output dir")
            os.mkdir(self.out_dir_path)

    def initialize_file_list(self):

        self.in_file_paths = []
        for root, dirs, files in os.walk(self.in_dir_path):
            for file in files:
                self.in_file_paths.append(root + '/' + file)

        self.out_file_paths = [path.replace(self.in_dir_path,self.out_dir_path) for path in self.in_file_paths]

    def txt_files(self,write_to_one_file=False):

        def ALFNet_detections_to_MOT(line,frame_id):

            line_elems = (line.split("\n")[0]).split(" ")
            line = " ".join([str(frame_id), '-1' , line_elems[0].split(".")[0] , line_elems[1].split(".")[0], line_elems[2].split(".")[0] , line_elems[3].split(".")[0] ,line_elems[4], '-1', '-1\n'])

            return line

        in_txt_file_paths = [txt_file_path for txt_file_path in self.in_file_paths if txt_file_path.endswith('.txt')]
        out_txt_file_paths = [txt_file_path for txt_file_path in self.out_file_paths if txt_file_path.endswith('.txt')]
        open_option = 'w'

        if write_to_one_file:
            print("Writing to one file named: " + out_txt_file_paths[0])
            out_txt_file_paths = [out_txt_file_paths[0] for _ in out_txt_file_paths]
            open_option = 'a'

        for frame_id,(in_txt_file_path,out_txt_file_path) in enumerate(zip(in_txt_file_paths,out_txt_file_paths)):

            with open(in_txt_file_path) as read_txt_file:
                with open(out_txt_file_path,open_option) as write_txt_file:

                    line = read_txt_file.readline()
                    while line:

                        line = ALFNet_detections_to_MOT(line,frame_id+1)

                        write_txt_file.write(line)
                        line = read_txt_file.readline()

            if frame_id==10:
                exit()

if __name__ == "__main__":

    convert = Convert('/raid/algo/SOCISP_SLOW/ADAS/PD/Cityscapes/results/inference/AlfNet/mobilenet/demo/stuttgart_for_OT_mobilenet_2step_c0.30',
                      '/raid/algo/SOCISP_SLOW/ADAS/PD/Cityscapes/results/inference/AlfNet/mobilenet/demo/stuttgart_for_OT_mobilenet_2step_c0.30/temp')

    convert.initialize_file_list()

    convert.txt_files(write_to_one_file=True)
```

### Python crash course

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