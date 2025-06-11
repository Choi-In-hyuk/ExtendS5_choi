python run_train.py --C_init=lecun_normal --batchnorm=True --bidirectional=False \
                    --blocks=3 --bsz=50 --clip_eigs=True --d_model=512 --dataset=lra-cifar-classification \
                    --epochs=250 --jax_seed=1641641 --lr_factor=4.5 --n_layers=6 --opt_config=BfastandCdecay \
                    --p_dropout=0.1 --ssm_lr_base=0.001 --ssm_size_base=126 --warmup_end=1 --weight_decay=0.07 \
                    --USE_WANDB=True --discretization=bilinear --R=126 --conj_sym=False