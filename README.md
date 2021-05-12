# LReasoner
The source code of Paper "Logic-Driven Context Extension and Data Augmentation for Logical Reasoning of Text"

Our ensemble system is the **first to surpass human performance on both EASY set and HARD set of ReClor** ([EvalAI leaderboard](https://evalai.cloudcv.org/web/challenges/challenge-page/503/leaderboard/1347)). If you find this paper useful, please cite this paper:
```
@misc{wang2021logicdriven,
      title={Logic-Driven Context Extension and Data Augmentation for Logical Reasoning of Text}, 
      author={Siyuan Wang and Wanjun Zhong and Duyu Tang and Zhongyu Wei and Zhihao Fan and Daxin Jiang and Ming Zhou and Nan Duan},
      year={2021},
      eprint={2105.03659},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
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
1. to run LReasoner_roberta
        bash run_roberta_DA_CE.sh
    
2. to run LReasoner_albert
        bash run_albert_DA_CE.sh
```
Here **CE** means context extension while **DA** means data augmentation.


## Result
We obtain the following results:

|  Model   | Test | EASY | HARD |
|  ----  | ----  | ----  |  ----  |
|  LReasoner-RoBERTa  |  62.4  |  81.4  |  47.5  |
|  LReasoner-ALBERT  |  70.7  |  81.1  |  62.5  |
|  Human Performance  |  63.0  |  57.1  |  67.2  |

We only evaluate our model on the ReClor dataset and plan to try more datasets. As codes are publicly available, anyone interested could also have a try. 
