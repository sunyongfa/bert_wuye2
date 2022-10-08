python train.py \
    --model_config_file='configs/configsenti/bert_cls.json' \
    --save_model_path='model_saved/model_senti/bert/bert_test' \
    --run_mode=eval \
    --eval_data='./data/sentiment/test1/test.tsv' \
    --batch_size=32 \
    --data_load_num_workers=2 \
    --gpu_ids=-1 \
    --debug_break=0
