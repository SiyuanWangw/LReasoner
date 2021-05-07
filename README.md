# LReasoner
The source code of Paper "Logic-Driven Context Extension and Data Augmentation for Logical Reasoning of Text"

Our ensemble system is the **first to surpass human performance on both EASY set and HARD set of ReClor** ([EvalAI leaderboard](https://evalai.cloudcv.org/web/challenges/challenge-page/503/leaderboard/1347)). If you find this paper useful, please cite this paper:
```
@
```

## Setting up
1. To set up the environment, please install the packages in the `requirements.txt`.
```bash
pip install -r requirements.txt
```

2. To get the datasets, you can refer to the paper [ReClor: A Reading Comprehension Dataset Requiring Logical Reasoning](https://openreview.net/pdf?id=HJgJtT4tvB) to get the original data. We also provide our preprocessed data in `reclor-data` directory. Or you can start from the original data and preprocess it by the following steps:
 * Step 1: exrtact the logical symbols and identify the logical expressions in the context, then infer the entailed logical expressions;
 * Step 2: select a logical expreesion in the context and construct the negative samples based on it.
```bash
cd DataPreprocess
python extract_logical_expressions_v2.py
python construct_negative_samples_v2.py
```

## Usage
Then you can run the LReasoner system in the scripts directory as following:
```bash
1. for LReasoner_roberta
        bash run_roberta_DA_CE.sh
    
2. for LReasoner_albert
        bash run_albert_DA_CE.sh
```
Here **CE** means context extension while **DA** means data augmentation.


## Result
We obtain the following results:

|  Model   | Val  | Test | Test-E | Test-H |
|  ----  | ----  |  ----  | ----  |  ----  |
|  bert-base  | 54.6  |  47.3 | 71.6 |  28.2  |
|  bert-large  | 53.8  |  49.8  | 72.0  |  32.3  |
|  xlnet-base  | 55.8  |  50.4  | 75.2  |  32.9  |
|  xlnet-large  | 62.0  |  56.0 | 75.7  |  40.5  |
|  roberta-base  | 55.0  |  48.5  | 71.1  |  30.7  |
|  roberta-large  | 62.6  |  55.6  | 75.5  |  40.0  |
