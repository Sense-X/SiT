work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:../../ \
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --data-path your_path_to_data \
    --batch-size 128 \
    --epoch 125 \
    --lr 2e-4 \
    --warmup-epochs 0 \
    --distillation-type frd \
    --lambda-token 2.0 --lambda-logits 2.0 \
    --input-size 288 --drop-path 0.3 --aa rand-n3-m9-mstd0.5-inc1 \
    --model sit_lvvit_large \
    --finetune your_path_to_teacher_model \
    --teacher-model sit_lvvit_large \
    --teacher-path your_path_to_teacher_model \
    --dist-eval \
    --output_dir ${work_path}/ckpt \
    2>&1 | tee -a ${work_path}/log.txt

