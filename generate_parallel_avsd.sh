# export MODEL=$1
# export TAG=$2
# export MODE=$3
# export EVAL_DIR=$4
# export MEDIUM=$5
# export DSTC=$6

export MODEL='v2dial/stage_3'
export TAG='finetuned_no_experts_avsd'
export MODE='generate'
export EVAL_DIR='/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/ma35vahy/V2Dial_new_v2/logs/stage_3/v2dial-google_flan-t5-large-finetune_without_experts_avsd'
export DSTC=7

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate v2dial

if [ $DSTC -eq 10 ]; then
    export CUDA_VISIBLE_DEVICES=0; python main_stage_3.py --start_idx_gen 0000 --end_idx_gen 0112 --gen_subset_num 01 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
    export CUDA_VISIBLE_DEVICES=1; python main_stage_3.py --start_idx_gen 0112 --end_idx_gen 0224 --gen_subset_num 02 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
    export CUDA_VISIBLE_DEVICES=2; python main_stage_3.py --start_idx_gen 0224 --end_idx_gen 0336 --gen_subset_num 03 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=3; python main_stage_3.py --start_idx_gen 0336 --end_idx_gen 0448 --gen_subset_num 04 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
    export CUDA_VISIBLE_DEVICES=4; python main_stage_3.py --start_idx_gen 0448 --end_idx_gen 0560 --gen_subset_num 05 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=5; python main_stage_3.py --start_idx_gen 0560 --end_idx_gen 0672 --gen_subset_num 06 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=6; python main_stage_3.py --start_idx_gen 0672 --end_idx_gen 0784 --gen_subset_num 07 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=7; python main_stage_3.py --start_idx_gen 0784 --end_idx_gen 0896 --gen_subset_num 08 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=0; python main_stage_3.py --start_idx_gen 0896 --end_idx_gen 1008 --gen_subset_num 09 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
    export CUDA_VISIBLE_DEVICES=1; python main_stage_3.py --start_idx_gen 1008 --end_idx_gen 1120 --gen_subset_num 10 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
    export CUDA_VISIBLE_DEVICES=2; python main_stage_3.py --start_idx_gen 1120 --end_idx_gen 1232 --gen_subset_num 11 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=3; python main_stage_3.py --start_idx_gen 1232 --end_idx_gen 1344 --gen_subset_num 12 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
    export CUDA_VISIBLE_DEVICES=4; python main_stage_3.py --start_idx_gen 1344 --end_idx_gen 1456 --gen_subset_num 13 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=5; python main_stage_3.py --start_idx_gen 1456 --end_idx_gen 1568 --gen_subset_num 14 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=6; python main_stage_3.py --start_idx_gen 1568 --end_idx_gen 1680 --gen_subset_num 15 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=7; python main_stage_3.py --start_idx_gen 1680 --end_idx_gen 1804 --gen_subset_num 16 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
else
    export CUDA_VISIBLE_DEVICES=0; python main_stage_3.py --start_idx_gen 0000 --end_idx_gen 0107 --gen_subset_num 01 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
    export CUDA_VISIBLE_DEVICES=1; python main_stage_3.py --start_idx_gen 0107 --end_idx_gen 0214 --gen_subset_num 02 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
    export CUDA_VISIBLE_DEVICES=2; python main_stage_3.py --start_idx_gen 0214 --end_idx_gen 0321 --gen_subset_num 03 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=3; python main_stage_3.py --start_idx_gen 0321 --end_idx_gen 0428 --gen_subset_num 04 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
    export CUDA_VISIBLE_DEVICES=4; python main_stage_3.py --start_idx_gen 0428 --end_idx_gen 0535 --gen_subset_num 05 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=5; python main_stage_3.py --start_idx_gen 0535 --end_idx_gen 0642 --gen_subset_num 06 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=6; python main_stage_3.py --start_idx_gen 0642 --end_idx_gen 0749 --gen_subset_num 07 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=7; python main_stage_3.py --start_idx_gen 0749 --end_idx_gen 0856 --gen_subset_num 08 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=0; python main_stage_3.py --start_idx_gen 0856 --end_idx_gen 0963 --gen_subset_num 09 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
    export CUDA_VISIBLE_DEVICES=1; python main_stage_3.py --start_idx_gen 0963 --end_idx_gen 1070 --gen_subset_num 10 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
    export CUDA_VISIBLE_DEVICES=2; python main_stage_3.py --start_idx_gen 1070 --end_idx_gen 1177 --gen_subset_num 11 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=3; python main_stage_3.py --start_idx_gen 1177 --end_idx_gen 1284 --gen_subset_num 12 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
    export CUDA_VISIBLE_DEVICES=4; python main_stage_3.py --start_idx_gen 1284 --end_idx_gen 1391 --gen_subset_num 13 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=5; python main_stage_3.py --start_idx_gen 1391 --end_idx_gen 1498 --gen_subset_num 14 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=6; python main_stage_3.py --start_idx_gen 1498 --end_idx_gen 1605 --gen_subset_num 15 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
    export CUDA_VISIBLE_DEVICES=7; python main_stage_3.py --start_idx_gen 1605 --end_idx_gen 1710 --gen_subset_num 16 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
fi

# export CUDA_VISIBLE_DEVICES=6; python main_stage_3.py --start_idx_gen 00 --end_idx_gen 10 --gen_subset_num 15 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
# export CUDA_VISIBLE_DEVICES=7; python main_stage_3.py --start_idx_gen 10 --end_idx_gen 20 --gen_subset_num 16 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 

wait
python merge_pred_avsd.py --dstc $DSTC
