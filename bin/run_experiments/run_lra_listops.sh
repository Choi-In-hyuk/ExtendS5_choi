python run_train.py --C_init=lecun_normal --activation_fn=half_glu2 --batchnorm=True \
                    --bidirectional=False --blocks=8 --bsz=50 --d_model=128 --dataset=listops-classification \
                    --epochs=50 --jax_seed=65 --lr_factor=3 --n_layers=6 --opt_config=BfastandCdecay \
                    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=16 --warmup_end=1 --weight_decay=0.04 --R=16