python run_train.py --C_init=lecun_normal --batchnorm=True --bidirectional=False \
                    --blocks=16 --bsz=16 --d_model=96 --dataset=selective-copying \
                    --epochs=40 --jax_seed=4062966 --lr_factor=4 --n_layers=6 --opt_config=noBCdecay \
                    --p_dropout=0.1 --ssm_lr_base=0.002 --ssm_size_base=128 --warmup_end=1 --weight_decay=0.04 \
                    --R=16 --ssm=False --USE_WANDB=False