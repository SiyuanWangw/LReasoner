export TASK_NAME=reclor
export RECLOR_DIR=../reclor-data


# RoBERTa+DA+CE model
python -m torch.distributed.launch --nproc_per_node=4 main_large_contrastive.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name $TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --do_test \
    --do_lower_case \
    --data_dir $RECLOR_DIR \
    --max_seq_length 288 \
    --per_gpu_eval_batch_size 2  \
    --per_gpu_train_batch_size 2  \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 10.0 \
    --output_dir ../Checkpoints/reclor/roberta_augmentation_extension \
    --logging_steps 200 \
    --save_steps 200 \
    --adam_betas "(0.9, 0.98)" \
    --adam_epsilon 1e-6 \
    --no_clip_grad_norm \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --overwrite_output_dir \
    --ques_type_before 1 \
    --overwrite_cache \
    --extended_context_version 5 \
    --negative_context_version 19 \
    --negative_entend_context_version 195 \
    --local_rank 0 \
    --fp16 \
    --seed 4321 \
    --whether_extend_context \



