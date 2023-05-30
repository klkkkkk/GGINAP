# GGINAP
## Quick start
### Step 1: Required environment
```
python           3.10
pytorch          1.12.1+cu113
dgl              1.0.1+cu113
transformers     4.26.1
scipy            1.10.1
scikit-learn     1.2.1
```
Other version may also work.
### Step 2: Download the Bert-base-chinese
Download the Bert-base-chinese from [here](https://huggingface.co/bert-base-chinese). Then put the files to
```
./pretrained_model/bert-base-chinese/
```

### Step 3: Run the model
```
python main.py
```
