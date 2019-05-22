# Unification of DB  annotations and metadata

Each DB annotations and metadata will be present in one data structure holding all the data required, this will be comprised out of two pieces, the metadata which hold DB specific information and the data which holds object specific information. 

Note that the DBs chosen here are already split in to Train/Test/Validation

An example of construction of this kind of data structure is at: 							'PedestrianDetection/SSIC-ATG_DB_Visualization/Statistics/Creating_DF_for_CS.ipynb'

## Access

```python
import pandas as pd

def h5store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('mydata', df)
    store.get_storer('mydata').attrs.metadata = kwargs
    store.close()

def h5load(store):
    data = store['mydata']
    metadata = store.get_storer('mydata').attrs.metadata
    return data, metadata

with pd.HDFStore(path_to_datastructure) as store:
    data, metadata = h5load(store)
```

## Metadata

Metadata holds DB specific information.

- db_name - name of the Data base 
- imgs_path - path to image directory
- cat_dict - a category dictionary which returns a label for category number, it is DB specific
- seg_gt_path - path to ground truth segmentation data 
- image_width / image_height - image width / height in px

As an example of metadata initialization and access:

```python
# see definition of car_dict under the dictionary section below.

db_metadata = {'db_name':'Citypersons',
               'imgs_path':'/mnt/algo-			  datasets/DB/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val',
                'seg_gt_path': '/mnt/datasets/DB/Cityscapes/Fuller_GT/val',
                'cat_dict': cat_dict,
                'img_width':2048,
                'img_height':1024}

>>> metadata['img_height']
1024
```

## Data

Data holds object and image specific information.

- BB.x / BB.y / BB.w / BB.h - bounding boxes top left corner, width and height
- BB_V.x / BB_V.y / BB_V.w / BB_V.h - visible bounding boxes top left corner, width and height
- img_id - image id (Only cityscapes!)
- cat_id - category id of each object (DB specific, see below)
- frame_rpath - path to frame relative to it's folder
- dir_rpath - path relative to 'db_path' which is in the metadata to the frames 
- to
- be
- followed
- with 
- SSIC-ATG 
- stuff

## Dictionary

Since different annotations are given to different DBs, every data base will have it's own category dictionary. In order to solve this issue, a dictionary is attached in the metadata and used as follows:

```python
cat_dict = metadata['cat_dict']

>>> cat_map[data['cat_id'][0]]
'sitting person'
```

In Citypersons, the dictionary is build as such:

```python
cat_dict = {0:'ignore',
            1:'pedestrian',
            2:'rider',
            3:'sitting person',
            4:'other person',
            5:'person group'}
```





