# S5 Finetuning & ExtendS5 실행 가이드 
# S5 finetuning 코드 실행 방법 
## 기본 파인튜닝 (freeze 없이) 
```bash
python choi_run_train.py --dataset=imdb-classification --checkpoint=True --USE_WANDB=False --n_layers=6 --d_model=256 --ssm_size_base=192 --blocks=12 --C_init=lecun_normal --discretization=zoh --mode=pool --activation_fn=half_glu2 --conj_sym=True --clip_eigs=True --bidirectional=True --dt_min=0.001 --dt_max=0.1 --prenorm=True --batchnorm=True --bn_momentum=0.95 --bsz=50 --epochs=35 --early_stop_patience=1000 --ssm_lr_base=0.001 --lr_factor=4.0 --dt_global=True --lr_min=0 --cosine_anneal=True --warmup_end=0 --lr_patience=1000000 --reduce_factor=1.0 --p_dropout=0.1 --weight_decay=0.07 --opt_config=standard
```
## layers 1,2의 A,C 파라미터 freeze
```bash
python choi_run_train.py --dataset=imdb-classification --checkpoint=True --USE_WANDB=False --n_layers=6 --d_model=256 --ssm_size_base=192 --blocks=12 --C_init=lecun_normal --discretization=zoh --mode=pool --activation_fn=half_glu2 --conj_sym=True --clip_eigs=True --bidirectional=True --dt_min=0.001 --dt_max=0.1 --prenorm=True --batchnorm=True --bn_momentum=0.95 --bsz=50 --epochs=35 --early_stop_patience=1000 --ssm_lr_base=0.001 --lr_factor=4.0 --dt_global=True --lr_min=0 --cosine_anneal=True --warmup_end=0 --lr_patience=1000000 --reduce_factor=1.0 --p_dropout=0.1 --weight_decay=0.07 --opt_config=standard --freeze_layers=1,2 --freeze_params=A,C
```
## 하위 레이어들(0,1,2)의 모든 SSM 파라미터 freeze
```bash
python choi_run_train.py --dataset=imdb-classification --checkpoint=True --USE_WANDB=False --n_layers=6 --d_model=256 --ssm_size_base=192 --blocks=12 --C_init=lecun_normal --discretization=zoh --mode=pool --activation_fn=half_glu2 --conj_sym=True --clip_eigs=True --bidirectional=True --dt_min=0.001 --dt_max=0.1 --prenorm=True --batchnorm=True --bn_momentum=0.95 --bsz=50 --epochs=35 --early_stop_patience=1000 --ssm_lr_base=0.001 --lr_factor=4.0 --dt_global=True --lr_min=0 --cosine_anneal=True --warmup_end=0 --lr_patience=1000000 --reduce_factor=1.0 --p_dropout=0.1 --weight_decay=0.07 --opt_config=standard --freeze_layers=0,1,2 --freeze_params=A,B,C,D
```
# ExtendS5 실행 명령어
## S5 체크포인트 자동 로드 + freeze 적용
```bash
python -m s5.choi_train_ex --ssm_type extend --dataset lra-cifar-classification --R 10 --freeze_layers 0,1,2 --freeze_params A,B,C --d_model 256 --n_layers 6
``` 
## 다른 체크포인트 경로 사용
```bash
python -m s5.chio_train_ex --ssm_type extend --dataset lra-cifar-classification --R 10 --load_s5_checkpoint /home/choi/ExtendS5/checkpoints/imdb-classification/model_epoch_001.ckpt --d_model 256 --n_layers 6
```


