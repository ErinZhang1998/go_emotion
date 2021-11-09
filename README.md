# stress-multitask

## Environment
Please install python 3.8 and pytorch 1.6. You'll also need 
- ax
- huggingface transformers
- numpy
- scikit-learn

In order to run LIME explanation generation, you will also need
- lime

## Usage

#### 11-711 Training
```
for seed in 46921 25577 52538 56440 41228 66558 48642 69556
do
   template="CUDA_VISIBLE_DEVICES=4,5 python main.py train --main_dataset /raid/xiaoyuz1/goemotions/goemotions/data/train.csv --dev_file /raid/xiaoyuz1/goemotions/goemotions/data/dev.csv --test_file /raid/xiaoyuz1/goemotions/goemotions/data/test.csv --model bert --epochs 20 --data /raid/xiaoyuz1/goemotions/goemotions/data/test.csv --training_seed ${seed} --save_path /raid/xiaoyuz1/goemotions/save_path/bert_seed-${seed} --out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_seed-${seed}.txt"
   eval $template
   template2="CUDA_VISIBLE_DEVICES=4,5 python main.py predict --model_path /raid/xiaoyuz1/goemotions/save_path/bert_seed-${seed} --data /raid/xiaoyuz1/goemotions/goemotions/data/test.csv --out_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_debug.txt --out_label_path /raid/xiaoyuz1/goemotions/pred_result/baseline/test_pred_seed-${seed}_label.pkl"
   eval $template2
done
```

#### Training

The main entry point for this code is `main.py`. You can run training as follows:

```
python main.py train --main_dataset path/to/train/fn --dev_file path/to/dev/fn --test_file path/to/test/fn --model model_name
```

The train, development, and test files should all be .csv files with two columns: text and then labels.

Various additional train flags and arguments may be helpful:
- `--optimize`, which will use the `ax` library to optimize parameters.
- `--save_path`, for saving the model. The code will append `-params.pth` onto this to save the model, and will also 
save meta-information, like the label2idx associations and test scores, with `-meta.pkl`.
- `--save_lm`, which will save the language model (`BertModel`) only. This requires `--save-path` and will append `-lm` 
to the path. This is useful for sequential training scripts, such as our Fine-Tune models.
- `--log`, to save the log to a file as well as print it to stdout, and `--logfile`, to specify the log location.

**Multitask learning** is supported by the `--aux_datasets` argument, which should be given a list of files in the same 
format as the train file.

TODO: add training commands in `bin/` after results are finalized.

#### Prediction

You can run prediction using a saved model as follows:

```
python main.py predict --model_path path/to/model --data path/to/data/to/label --out_file path/to/save/data
```

The data to label must be in the same format as the training data (two columns, with headers).

Provide the `--model_path` in the same format as the `--save_path` for training, without any `-params.pth` suffix, 
because the code will need to load both the model parameters and the pickled meta-information.

#### Analysis

You can run analysis from the paper using `interpret_lime.py` and then `interpret_liwc.py`. These scripts expect certain
models to be saved in a `saved_models/` directory; you may change these model names in the relevant loops. Create a 
`lime/` directory in the top level of this repository before running either.

`interpert_liwc.py` also expects a LIWC word list to be present as `data/liwc.tsv`. We cannot share our copy of this 
file, but it was created using LIWC 2015 and is in `.tsv` format, with one column 'Category' and another 'Words'. 
'Words' is a space-delimited list of strings from LIWC, where an asterisk (*) is included in some words to denote that 
any ending is accepted (e.g., "walk*" would accept "walking", "walks", and "walked" as well as "walk").