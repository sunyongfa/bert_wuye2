#!/bin/sh 


#预警配置文件
common='configs/configwarn/'
common1='bert_cls.json'
common2='bert_cls_ernie.json'
common3='bert_cls_nezha_base.json'
common4='bert_cls_roberta_wwm_ext.json'
common5='bert_cls_nezha_base_wwm.json'
common1=$common$common1
common2=$common$common2
common3=$common$common3
common4=$common$common4
common5=$common$common5
config_warn_files=($common1 $common2 $common3 $common4 $common4)



#预警全部数据训练 模型存放路径
path7='data/warning/all/train.tsv'
path8='data/warning/all/dev.tsv'

save_model_warn_path='model_saved/model_warn/'
s1='bert'
s2='ernie'
s3='nezha_base'
s4='roberta_wwm_ext'
s5='nezha_base_wwm'
s='/all/bert_test'
s1=$save_model_warn_path$s1$s
s2=$save_model_warn_path$s2$s
s3=$save_model_warn_path$s3$s
s4=$save_model_warn_path$s4$s
s5=$save_model_senti_path$s5$s
save_model_allwarn_paths=($s1 $s2 $s3 $s4 $s5)

save_model_warn_path='model_saved/model_warn/'
s1='bert'
s2='ernie'
s3='nezha_base'
s4='roberta_wwm_ext'
s5='nezha_base_wwm'
s='/all/bert_test_distill'
s1=$save_model_warn_path$s1$s
s2=$save_model_warn_path$s2$s
s3=$save_model_warn_path$s3$s
s4=$save_model_warn_path$s4$s
s5=$save_model_senti_path$s5$s
save_model_allwarn_path_distills=($s1 $s2 $s3 $s4 $s5)


train1(){
    
    p1=$1
	p2=$2
	
	local -n config_file=$3
	local -n save_model_path=$4
	local -n save_model_path_ditill=$5
	
	n1=$6
	n2=$7
	
    for i in 0 1 2 3 4
    do
	   file1=${config_file[i]} 
	   file2=${save_model_path[i]} 
	   file3=${save_model_path_ditill[i]}
	   echo $file1 $file2 $file3
	   
       python train.py \
           --model_config_file=$file1 \
           --save_model_path=$file2 \
            --run_mode=train \
            --train_stage=0 \
            --train_data=$p1 \
            --eval_data=$p2 \
            --epochs=$n1 \
            --batch_size=32 \
            --data_load_num_workers=2 \
            --gpu_ids=-1 \
            --debug_break=0
		python train.py \
                --model_config_file=$file1 \
                --save_model_path=$file2 \
                --save_model_path_distill=$file3 \
                --run_mode=train \
                --train_stage=1 \
                --train_data=$p1 \
                --eval_data=$p2 \
                --epochs=$n2 \
                --batch_size=32 \
                --data_load_num_workers=2 \
                --gpu_ids=-1 \
                --debug_break=0
        
	done


}


train1 $path7 $path8 config_warn_files save_model_allwarn_paths save_model_allwarn_path_distills 14 14




<<COMMENT
#预警配置文件
common='configs/configwarn/'
common6='bert_cls_roberta_wwm_ext_large.json'
common7='bert_cls_nezha_large.json'
common8='bert_cls_nezha_large_wwm.json'

common6=$common$common6
common7=$common$common7
common8=$common$common8

config_warn_files=($common6  $common7 $common8)




#情绪需要测试的数据 模型存放路径
path7='data/warning/all/train.tsv'
path8='data/warning/all/dev.tsv'


save_model_warn_path='model_saved/model_warn/'
s6='roberta_wwm_ext_large'
s7='nezha_large'
s8='nezha_large_wwm'

s='/all/bert_test'
s6=$save_model_senti_path$s6$s
s7=$save_model_senti_path$s7$s
s8=$save_model_senti_path$s8$s

save_model_warn_paths=($s6 $s7 $s8 )

save_model_warn_path='model_saved/model_warn/'
s6='roberta_wwm_ext_large'
s7='nezha_large'
s8='nezha_large_wwm'
s='/all/bert_test_distill'
s6=$save_model_senti_path$s6$s
s7=$save_model_senti_path$s7$s
s8=$save_model_senti_path$s8$s

save_model_warn_path_ditills=($s6 $s7 $s8)



train2(){
    
    p1=$1
	p2=$2
	
	local -n config_file=$3
	local -n save_model_path=$4
	local -n save_model_path_ditill=$5
	
	n1=$6
	n2=$7
	
    for i in 0 1 2
    do
	   file1=${config_file[i]} 
	   file2=${save_model_path[i]} 
	   file3=${save_model_path_ditill[i]}
	   echo $file1 $file2 $file3
	   
       python train.py \
           --model_config_file=$file1 \
           --save_model_path=$file2 \
            --run_mode=train \
            --train_stage=0 \
            --train_data=$p1 \
            --eval_data=$p2 \
            --epochs=$n1 \
            --batch_size=32 \
            --data_load_num_workers=2 \
            --gpu_ids=-1 \
            --debug_break=0
		python train.py \
                --model_config_file=$file1 \
                --save_model_path=$file2 \
                --save_model_path_distill=$file3 \
                --run_mode=train \
                --train_stage=1 \
                --train_data=$p1 \
                --eval_data=$p2 \
                --epochs=$n2 \
                --batch_size=32 \
                --data_load_num_workers=2 \
                --gpu_ids=-1 \
                --debug_break=0
        
	done


}


train2 $path7 $path8 config_warn_files save_model_allwarn_paths save_model_allwarn_path_distills 14 14

COMMENT