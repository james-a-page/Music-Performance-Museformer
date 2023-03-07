cd museformer

mkdir -p data-bin/instruct_museformer

fairseq-preprocess --source-lang score --target-lang perf --trainpref ../data/tokens/train --validpref ../data/tokens/val --testpref ../data/tokens/test --destdir data-bin/instruct_museformer --srcdict data/meta/our_dict.txt --tgtdict data/meta/our_dict.txt

cp data/meta/our_dict.txt data-bin/instruct_museformer/dict.txt

fairseq-train data-bin/instruct_museformer --user-dir museformer --task museformer_language_modeling --arch museformer_lm_v2s1 --con2con '((((-2, 0), -4, -8, -12, -16, -24, -32),),)' --con2sum '((((None, -32), (-31, -24), (-23, -16), (-15, -12), (-11, -8), (-7, -4), -3,),),)' --num-layers 4 --tokens-per-sample 100000 --truncate-train 15360 --truncate-valid 10240 --batch-size 1 --update-freq 1 --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-9 --weight-decay 0.01 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 16000   --max-update 1000000 --validate-interval 1000000000 --save-interval 1000000000 --save-interval-updates 5000 --fp16 --log-interval 10 --tensorboard-logdir tb_log/museformer_instruct  --num-workers 8 --save-dir checkpoints/museformer_instruct --beat-mask-ts True --take-bos-as-bar True --log-format simple | tee log/museformer_instruct.log