# LReasoner
The source code of Paper "Logic-Driven Context Extension and Data Augmentation for Logical Reasoning of Text"

If you find this paper useful, please cite this paper:
```
@
```

## Setting up
1. To set up the environment, please install the packages in the `requirements.txt`.
```
pip install -r requirements.txt
```

2. To get the datasets, you can refer to the paper [ReClor: A Reading Comprehension Dataset Requiring Logical Reasoning](https://openreview.net/pdf?id=HJgJtT4tvB). We also provide our preprocessed data in `reclor-data` folder. You can preprocess the data as following:
```
cd DataPreprocess
python extract_logical_expressions_v2.py
python construct_negative_samples_v2.py
```

## Usage


## Result
