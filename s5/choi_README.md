# 수정한 내용
choi_train.py와 choi_helpers.py 수정해서 원하는 레이어의 A,B,C,D freeze하고 실행하도록 해봤음  
choi_run_train.py로 실행하도록  
chio_train_ex.py로 니가 학습시켜놓은 체크포인트의 ABCD불러오고 나머지 파라미터들은 랜덤 초기화해서 학습하도록 해놨음  
학습은 되는데 니가 원하는대로 잘 되는지는 몰라서 말해주면 또 어떻게 수정해보도록함  
최대한 니 코드에서 수정한거라 보기 편할거임  


# S5 finetuning 코드 실행 방법 
 
```bash
# 기본 파인튜닝 (freeze 없이)
python choi_run_train.py --dataset=imdb-classification --checkpoint=True --USE_WANDB=False --n_layers=6 --d_model=256 --ssm_size_base=192 --blocks=12 --C_init=lecun_normal --discretization=zoh --mode=pool --activation_fn=half_glu2 --conj_sym=True --clip_eigs=True --bidirectional=True --dt_min=0.001 --dt_max=0.1 --prenorm=True --batchnorm=True --bn_momentum=0.95 --bsz=50 --epochs=35 --early_stop_patience=1000 --ssm_lr_base=0.001 --lr_factor=4.0 --dt_global=True --lr_min=0 --cosine_anneal=True --warmup_end=0 --lr_patience=1000000 --reduce_factor=1.0 --p_dropout=0.1 --weight_decay=0.07 --opt_config=standard

# layers 1,2의 A,C 파라미터 freeze
python choi_run_train.py --dataset=imdb-classification --checkpoint=True --USE_WANDB=False --n_layers=6 --d_model=256 --ssm_size_base=192 --blocks=12 --C_init=lecun_normal --discretization=zoh --mode=pool --activation_fn=half_glu2 --conj_sym=True --clip_eigs=True --bidirectional=True --dt_min=0.001 --dt_max=0.1 --prenorm=True --batchnorm=True --bn_momentum=0.95 --bsz=50 --epochs=35 --early_stop_patience=1000 --ssm_lr_base=0.001 --lr_factor=4.0 --dt_global=True --lr_min=0 --cosine_anneal=True --warmup_end=0 --lr_patience=1000000 --reduce_factor=1.0 --p_dropout=0.1 --weight_decay=0.07 --opt_config=standard --freeze_layers=1,2 --freeze_params=A,C

# 하위 레이어들(0,1,2)의 모든 SSM 파라미터 freeze
python choi_run_train.py --dataset=imdb-classification --checkpoint=True --USE_WANDB=False --n_layers=6 --d_model=256 --ssm_size_base=192 --blocks=12 --C_init=lecun_normal --discretization=zoh --mode=pool --activation_fn=half_glu2 --conj_sym=True --clip_eigs=True --bidirectional=True --dt_min=0.001 --dt_max=0.1 --prenorm=True --batchnorm=True --bn_momentum=0.95 --bsz=50 --epochs=35 --early_stop_patience=1000 --ssm_lr_base=0.001 --lr_factor=4.0 --dt_global=True --lr_min=0 --cosine_anneal=True --warmup_end=0 --lr_patience=1000000 --reduce_factor=1.0 --p_dropout=0.1 --weight_decay=0.07 --opt_config=standard --freeze_layers=0,1,2 --freeze_params=A,B,C,D
```

# ExtendS5 실행 명령어

```bash
# S5 체크포인트 자동 로드 + freeze없이 실행
python -m s5.choi_train_ex --ssm_type extend --dataset imdb-classification --R 10 --d_model 256 --n_layers 6 --ssm_size_base 96 --epochs 50

# 모든 레이어의 ABCD freeze
python -m s5.choi_train_ex --ssm_type extend --dataset imdb-classification --R 10 \
  --freeze_layers=0,1,2,3,4,5 --freeze_params A,B,C,D \  
  --d_model 256 --n_layers 6 --ssm_size_base 96 --epochs 50 















