# export MODEL=$1
# export TAG=$2
# export MODE=$3
# export EVAL_DIR=$4

export MODEL='v2dial/stage_3'
export TAG='nextqa_with_test_captions'
export MODE='generate'
export EVAL_DIR='/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/ma35vahy/V2Dial_new_v2/logs/stage_3/v2dial-google_flan-t5-large-from_stage1_only_nextqa_after_avsd_4_frames_3_rounds_ft_fp16'

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

# export CUDA_VISIBLE_DEVICES=0; python main_stage_3.py --start_idx_gen 0000 --end_idx_gen 10 --gen_subset_num 01 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
# export CUDA_VISIBLE_DEVICES=0; python main_stage_3.py --start_idx_gen 10 --end_idx_gen 20 --gen_subset_num 02 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \


export CUDA_VISIBLE_DEVICES=0; python main_stage_3.py --start_idx_gen 0000 --end_idx_gen 0573 --gen_subset_num 01 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=0; python main_stage_3.py --start_idx_gen 0573 --end_idx_gen 1146 --gen_subset_num 02 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=1; python main_stage_3.py --start_idx_gen 1146 --end_idx_gen 1719 --gen_subset_num 03 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=1; python main_stage_3.py --start_idx_gen 1719 --end_idx_gen 2292 --gen_subset_num 04 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=2; python main_stage_3.py --start_idx_gen 2292 --end_idx_gen 2865 --gen_subset_num 05 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=2; python main_stage_3.py --start_idx_gen 2865 --end_idx_gen 3438 --gen_subset_num 06 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=3; python main_stage_3.py --start_idx_gen 3438 --end_idx_gen 4011 --gen_subset_num 07 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=3; python main_stage_3.py --start_idx_gen 4011 --end_idx_gen 4584 --gen_subset_num 08 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & 
export CUDA_VISIBLE_DEVICES=4; python main_stage_3.py --start_idx_gen 4584 --end_idx_gen 5157 --gen_subset_num 09 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=4; python main_stage_3.py --start_idx_gen 5157 --end_idx_gen 5730 --gen_subset_num 10 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=5; python main_stage_3.py --start_idx_gen 5730 --end_idx_gen 6303 --gen_subset_num 11 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=5; python main_stage_3.py --start_idx_gen 6303 --end_idx_gen 6876 --gen_subset_num 12 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=6; python main_stage_3.py --start_idx_gen 6876 --end_idx_gen 7449 --gen_subset_num 13 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=6; python main_stage_3.py --start_idx_gen 7449 --end_idx_gen 8022 --gen_subset_num 14 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=7; python main_stage_3.py --start_idx_gen 8022 --end_idx_gen 8495 --gen_subset_num 15 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=7; python main_stage_3.py --start_idx_gen 8495 --end_idx_gen 9178 --gen_subset_num 16 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \

wait

python merge_pred_nextqa.py