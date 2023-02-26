## Instructions/Notes

We're making use of the Museformer model from Muzic by Microsoft (https://github.com/microsoft/muzic/tree/main/museformer).

This uses fairseq (https://github.com/facebookresearch/fairseq) to run.
```cmd
pip install fairseq
```

We can skip their first step which describes the generation of tokens from MIDI files as we have our own generation approach with custom tokenisation process.

Therefore the first step is to save our tokenisations into .txt format into a data/split/{test,train,valid}.txt file.

We need a train.txt, test.txt and a val.txt each containing the token list on for each data point on a new line.




From their we will make use of their commands:
```cmd
split_dir=data/split
data_bin_dir=data-bin/lmd6remi

mkdir -p data-bin

fairseq-preprocess \
  --only-source \
  --trainpref $split_dir/train.data \
  --validpref $split_dir/valid.data \
  --testpref $split_dir/test.data \
  --destdir $data_bin_dir \
  --srcdict {insert path to our vocab}
```

```
pip install triton
```
