# export MODEL=$1
# export TAG=$2
# export MODE=$3
# export EVAL_DIR=$4
# export MEDIUM=$5
# export DSTC=$6

export MODEL='v2dial/stage_3'
export TAG='finetuned_visdial_from_scratch'
export MODE='generate'
export EVAL_DIR='/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/ma35vahy/V2Dial_new_v2/logs/stage_3/v2dial-google_flan-t5-large-finetuned_visdial_from_scratch/'

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
# export CUDA_VISIBLE_DEVICES=0; python main_stage_3.py --start_idx_gen 00000 --end_idx_gen 10 --gen_subset_num 01 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
# export CUDA_VISIBLE_DEVICES=1; python main_stage_3.py --start_idx_gen 00010 --end_idx_gen 20 --gen_subset_num 02 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \


export CUDA_VISIBLE_DEVICES=0; python main_stage_3.py --start_idx_gen 00000 --end_idx_gen 00645 --gen_subset_num 01 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=1; python main_stage_3.py --start_idx_gen 00645 --end_idx_gen 01290 --gen_subset_num 02 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=2; python main_stage_3.py --start_idx_gen 01290 --end_idx_gen 01935 --gen_subset_num 03 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=3; python main_stage_3.py --start_idx_gen 01935 --end_idx_gen 02580 --gen_subset_num 04 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=4; python main_stage_3.py --start_idx_gen 02580 --end_idx_gen 03225 --gen_subset_num 05 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=5; python main_stage_3.py --start_idx_gen 03225 --end_idx_gen 03870 --gen_subset_num 06 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=6; python main_stage_3.py --start_idx_gen 03870 --end_idx_gen 04515 --gen_subset_num 07 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=7; python main_stage_3.py --start_idx_gen 04515 --end_idx_gen 05160 --gen_subset_num 08 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=0; python main_stage_3.py --start_idx_gen 05160 --end_idx_gen 05805 --gen_subset_num 09 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=1; python main_stage_3.py --start_idx_gen 05805 --end_idx_gen 06450 --gen_subset_num 10 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=2; python main_stage_3.py --start_idx_gen 06450 --end_idx_gen 07095 --gen_subset_num 11 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=3; python main_stage_3.py --start_idx_gen 07095 --end_idx_gen 07740 --gen_subset_num 12 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=4; python main_stage_3.py --start_idx_gen 07740 --end_idx_gen 08385 --gen_subset_num 13 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=5; python main_stage_3.py --start_idx_gen 08385 --end_idx_gen 09030 --gen_subset_num 14 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=6; python main_stage_3.py --start_idx_gen 09030 --end_idx_gen 09675 --gen_subset_num 15 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=7; python main_stage_3.py --start_idx_gen 09675 --end_idx_gen 10320 --gen_subset_num 16 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=0; python main_stage_3.py --start_idx_gen 10320 --end_idx_gen 10965 --gen_subset_num 17 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=1; python main_stage_3.py --start_idx_gen 10965 --end_idx_gen 11610 --gen_subset_num 18 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=2; python main_stage_3.py --start_idx_gen 11610 --end_idx_gen 12255 --gen_subset_num 19 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=3; python main_stage_3.py --start_idx_gen 12255 --end_idx_gen 12900 --gen_subset_num 20 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=4; python main_stage_3.py --start_idx_gen 12900 --end_idx_gen 13545 --gen_subset_num 21 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=5; python main_stage_3.py --start_idx_gen 13545 --end_idx_gen 14190 --gen_subset_num 22 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=6; python main_stage_3.py --start_idx_gen 14190 --end_idx_gen 14835 --gen_subset_num 23 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=7; python main_stage_3.py --start_idx_gen 14835 --end_idx_gen 15480 --gen_subset_num 24 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=0; python main_stage_3.py --start_idx_gen 15480 --end_idx_gen 16125 --gen_subset_num 25 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=1; python main_stage_3.py --start_idx_gen 16125 --end_idx_gen 16770 --gen_subset_num 26 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=2; python main_stage_3.py --start_idx_gen 16770 --end_idx_gen 17415 --gen_subset_num 27 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=3; python main_stage_3.py --start_idx_gen 17415 --end_idx_gen 18060 --gen_subset_num 28 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \
export CUDA_VISIBLE_DEVICES=4; python main_stage_3.py --start_idx_gen 18060 --end_idx_gen 18705 --gen_subset_num 29 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=5; python main_stage_3.py --start_idx_gen 18705 --end_idx_gen 19350 --gen_subset_num 30 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=6; python main_stage_3.py --start_idx_gen 19350 --end_idx_gen 19995 --gen_subset_num 31 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 
export CUDA_VISIBLE_DEVICES=7; python main_stage_3.py --start_idx_gen 19995 --end_idx_gen 20640 --gen_subset_num 32 --model $MODEL --mode $MODE --eval_dir $EVAL_DIR --tag $TAG & \ 

wait
python eval_visdial.py
