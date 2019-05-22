# Useful python things

## Small things

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
input_file_paths = []
for root, dirs, files in os.walk(input_dir_path):
    path = root.split(os.sep)
    for file in files:
        input_file_paths.append(root + '/' + file)

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

