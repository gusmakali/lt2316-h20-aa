# LT2316 H20 Assignment A1

Name: Liliia Makashova

## Notes on Part 1.

PLEASE NOTE: the current implementation assumes that you will put data folder (`DDICorpus`) inside `aa` folder on the same level with data_loading.py and feature_extraction.py files. The `run.ipynb` should stay outside the `aa` folder. If you wish to arrange the files in another way, please change the path given as the `data_dir` paameter in `run.ipynb` respectively.

### parse_data function

This function uses two main functions to create the required dataframe - `create_data_df` and `create_ner_df` respectively.

---
***NOTE***

When `create_data_df` from `DataLoader` class runs for the first time, it will produce print statements that notify you about the progress in data parsing. Creating the `data_df` DataFrame takes longer than `ner_df`, as `data_df` is much bigger. Pickled `data_df` and `ner_df` will be used for consecutive runs to shorten execution time. No print statements are produced then.
---

Both of these main functions (`create_data_df` and `create_ner_df`) use their own helper functions for some smaller tasks. 

Design desicions:

- `vocab` and `id2word` - both essentially refer to the same list of unique tokens from data. 
- token ids - all unique tokens collected from data and assigned to an ID by their index in the vocabulary.`id2word`(that assings token_ids) has fisrt values as None so 0 will never be assigned to a token that exists in data.
- ner ids - 4 ner types collected in the array `[None, 'group', 'drug_n', 'drug', 'brand']` where None ensures that 0 is not assigned as an ID to any of NERs. Therefore resulting IDs - None=0, group=1, drug_n=2, drug=3, brand=4.
- the offsets that correspond to non sequence ners - e.g. 69-73;83-89 - are treated as two NERs for now and thus result in 2 entries in the `ner_df`
- the split of train data is made with consideration to not split it into two different dataframes - e.g. if `sentence0123` has 55 tokens, all of them will be assigned either to `TRAIN` or `VAL` splits, not for example 20 to `VAL` and 35 to `TRAIN`. I assumed that for training or validation it would be good to have full sentences with corresponding encoding. For this reason, `TEST` and `VAL` sets are not exacly the same length - `VAL` is a bit smaller - which shouldn't be a problem. The ratio might be changed during modeling, if it will be needed. 
- the 'Test for DDI Extraction' folder is used to collect only entities data (not pair_id data) for `test_df`. 
- the `get_y` function uses the `split_ner` function to split `ner_df` into three data frames (one for `TRAIN`, `TEST`, and `VAL`). Then `extract_tensor` combines each of those splits with similar `data_df` splits, extracts an array per sentence in each split and pads it to be of max sample length. It supports cutting to max sample length if that is max sample length is changed to a lower value.
- `max_samle_len` is determined by `discover_max_sample_len`. Currently it is 165 - the length of the longest sentence in the data. This value might be changed when training the model, as most sentences are of length between 15 and 35. So there might be no need to pad all tensors with accordance to the longest sentence. 
 
The returned dataframes are pickled and saved in the current directory to make the run time faster (pickling happens only on the first run of the `run.ipynb` notebook).

## Notes on Part 2.

### extract_features function

This function uses two helper functions - `add_features` and `encode_features`. Their names are very self explainatory.

Design desicions:

- as features I added 5 parameters (corresponds to 5 dim of the X_tensor) - POS-tag, word prefix(defined very arbitrarily), word_suffix(very arbitrarily as well), word length and the word itself (token_id). Possibly the number of features can be extended depening on the future model's performance. 

- the encoding of the features is made in the same way as the encoding of `token_id` from Part 1.

### DATA VISUALISATION

ALL PLOTS ARE SET TO .show(), NO FILES WITH PLOTS BEING CREATED.

dataset.plot_split_ner_distribution() - uses the return values from the `get_y` function. Plots ner distribution ignoring non-ner labels, as those are too many and the graph doesn't visually fit the data, obscuring the data for NER types. Shows that drug is the most common NER type along all splits.

## Notes on Part Bonus.

### dataset.plot_ner_per_sample_distribution() 
- uses the return values from `get_y`. The assignment says 'how many sentences has 1 ner, 2 ner and so on' so it was assumed by me, that you meant to say how many unique types of NER are in the sentences. E.g. if the padded array for one sentence looks like this [0,0,0,0,0,0,1,1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0] then the number of ner types here is 4 = two times of type 1, one time of types 2, 3, 4, and not ners (e.g 0) are disregarded. Therefore the value to plot here is not 5 (total count of non-zeros), but 4(unique ners count).

### dataset.plot_sample_length_distribution() 
- uses length of original (non-padded) sentences from `data_df`. Please note, you will see the biggest pick at around (x: 25, y: 400). This means that the biggest number of sentences (400) are of length 25 and they belong to a train set which is green (see plot's legend).

## dataset.plot_ner_cooccurence_venndiagram() 
- uses the pip installed `venn` module, creates sets of `ner_labels` over sentences in the whole `ner_df` data. 
