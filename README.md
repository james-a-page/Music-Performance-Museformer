## Instructions/Notes

We're making use of the Museformer model from Muzic by Microsoft (https://github.com/microsoft/muzic/tree/main/museformer).

This uses fairseq (https://github.com/facebookresearch/fairseq) to run make sure version == 0.10.2 as museformer appears incompatible with latest release. Install from the release page of github (if pip install not working.)

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
data_bin_dir=data-bin/instruct_museformer

mkdir -p data-bin/museformer

fairseq-preprocess --source-lang score --target-lang perf --trainpref data/tokens/train --validpref data/tokens/val --testpref data/tokens/test --destdir data-bin/instruct_museformer --srcdict data/meta/our_dict.txt --tgtdict data/meta/our_dict.txt
  ```



### Issues with museformer currently
Museformer is designed to work with a monolingual dataset, therefore the dataset processing stages try to initialse the dataset from one file per split. Their data pipeline goes through 5/6 processing stages.

TODO:
- Digest loader process
- Find process which we need to change to introduce the target data
- Change that process
- Test/hope/


```bash
fairseq-train data-bin/instruct_museformer --user-dir museformer --task museformer_language_modeling --arch museformer_lm_v2s1 --con2con '((((-2, 0), -4, -8, -12, -16, -24, -32),),)' --con2sum '((((None, -32), (-31, -24), (-23, -16), (-15, -12), (-11, -8), (-7, -4), -3,),),)' --num-layers 4 --tokens-per-sample 100000 --truncate-train 15360 --truncate-valid 10240 --batch-size 1 --update-freq 1 --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-9 --weight-decay 0.01 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 16000   --max-update 1000000 --validate-interval 1000000000 --save-interval 1000000000 --save-interval-updates 5000 --fp16 --log-interval 10 --tensorboard-logdir tb_log/museformer_instruct  --num-workers 8 --save-dir checkpoints/museformer_instruct --beat-mask-ts True --take-bos-as-bar True --log-format simple | tee log/museformer_instruct.log
```