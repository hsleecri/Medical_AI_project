# Active Learning project

This code is conducted in python = 3.10

# download the data from Kaggle

1. get your kaggle API as json file, and put it in the folder named, .kaggle
2. pip install kaggle in terminal
3. open terminal and type, 'kaggle datasets download -d subham07/detecting-anomalies-in-water-manufacturing'
4. If you successfully downloald the dataset, run data_loading.py

We used third-party library for Active Learning on Tabular Data

https://github.com/ValentinMargraf/ActiveLearningPipelines

```
pip install alpbench[full]
```

If error due to numpy dtype

# If error occurs due to numpy dtype

Manual mapping of numpy data types


dtype_mapping = {
    'int': [np.int8, np.int16, np.int32, np.int64],
    'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
    'float': [np.float16, np.float32, np.float64],
    'complex': [np.complex64, np.complex128],
    'bool': [np.bool_],
    'object': [np.object_],
    'str': [np.str_],
}

SIMPLE_NUMPY_TYPES = [
    nptype
    for type_cat, nptypes in dtype_mapping.items()
    for nptype in nptypes
    if type_cat != "others"
]

SIMPLE_TYPES = (bool, int, float, str, *SIMPLE_NUMPY_TYPES)
