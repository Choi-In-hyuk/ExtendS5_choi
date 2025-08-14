python run_train.py --C_init=lecun_normal --activation_fn=half_glu2 \
                    --batchnorm=True --bidirectional=True --blocks=3 --bsz=50 \
                    --d_model=512 --dataset=lra-cifar-classification \
                    --dt_global=True --epochs=250 --jax_seed=16416 --lr_factor=1 \
                    --n_layers=6 --opt_config=standard --p_dropout=0.1 --ssm_lr_base=0.001 \
                    --ssm_size_base=384 --warmup_end=0 --weight_decay=0.07 \
                    --checkpoint=True