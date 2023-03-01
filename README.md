## Instructions/Notes

We're making use of the Museformer model from Muzic by Microsoft (https://github.com/microsoft/muzic/tree/main/museformer).

This uses fairseq (https://github.com/facebookresearch/fairseq) to run.
```cmd
pip install fairseq
```

We can base our approach mainly off what they have done, but we will make some changes to work around our pipeline so far.

Firstly we already have a token generation process so we will skip the first step and generate a new instruct_dictionary.txt file using the same format as FairSeq requires. This format comprises of a .txt file with each line containing '{token_value} 1'.

This sets us up to continue using their workflow with some more modifications. Their training method uses the ```--only-source``` flag which means that they are training with a target being the same as the input trying to predict the next token in the sequence based on the input and the previous data. This is how they can generate new pieces with just a random or empty string as an input to the model. We want to learn with a performance as the target, so we will borrow from the design of language translation models that also use FairSeq and split our dataset into two files for each datapoint: {split}.perf and {split}.score where the .perf and .score act as 'language' flags which we want to learn a translation between. This requires that we replace the ```--only-source``` flag with ```--source-lang score --target-lang perf```.

We also make some changes to their model design in the token identifiers they point too such as end-of-bar which in their tokenisation is represented by ```b-1``` but we represent it as ```4```. We also need to change the default values given to the dictionary when it is loaded in CompoundDictionary.py, by default a dictionary in fairseq sets the ```bos```,```eos```, ```pad``` and ```unk``` tokens for us, but we have these defined separately in our tokeniser so we need to overwrite these.


With these changes, we should be good to then run ```fairseq-preprocess --flags...``` to encode our dataset in a readable format for the rest of the pipeline to run smoothly. 


### Additional Changes
- Change the fairseq end of bar token to be our bar token.
- https://fairseq.readthedocs.io/en/latest/command_line_tools.html#Preprocessing -> need to point the --source_lang --target_lang to an identifier for score vs the performance.
- test to see if above flags need to be added to the fairseq-train call or if fine as is after preprocessing

### Running

```powershell
split_dir=data/split
data_bin_dir=data-bin/lmd6remi

mkdir -p data-bin/lmd6remi

fairseq-preprocess --source-lang score --target-lang perf --trainpref data/split/train --validpref data/split/valid --testpref data/split/test --destdir data-bin/lmd6remi --srcdict data/meta/our_dict.txt
  ```
