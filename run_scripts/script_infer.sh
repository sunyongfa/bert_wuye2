python infer.py \
    --model_config_file='configs/configsenti/bert_cls.json' \
    --save_model_path='model_saved/model_senti/bert/test/bert_test_distill' \
    --inference_speed=0.1 \
    --infer_data='./data/sentiment/test1/test.tsv' \
    --dump_info_file='./data/sentiment/test1/infer_info.txt' \
    --data_load_num_workers=2 \
    --gpu_ids=-1 \
    --debug_break=0