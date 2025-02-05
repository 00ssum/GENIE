# GENIE
# Official code for 'Is large scale pretraining the secret to good domain generalization?'

To train CORAL:

```
python  train_all.py -m   coral_${test_env}_0  --data_dir=<data_path>  --algorithm CORAL --swad False   --test_envs $test_env --dataset $dataset  --seed 0 --trial_seed 0 --model openclip_vit-b16
```

To train SAGM:

```
python  train_all.py -m  sagm_${test_env}_0  --data_dir=<data_path>  --algorithm SAGM_DG --swad False    --test_envs $test_env --dataset $dataset  --seed 0 --trial_seed 0 --model openclip_vit-b16
```

To train ERM:

```
python  train_all.py -m  erm_${test_env}_0  --data_dir=<data_path>  --algorithm MIRO --swad False --ld 0.0   --test_envs $test_env --dataset $dataset  --seed 0 --trial_seed 0 --model openclip_vit-b16
```

To train LP-FT:

```
python train_all.py -m  lpft_${test_env}_0  --data_dir=<data_path>  --algorithm MIRO --swad False --ld 0.0   --test_envs $test_env --dataset $dataset  --seed 0 --trial_seed 0 --model openclip_vit-b16 --warmup
```

To train SWAD:

```
python  train_all.py -m swad_${test_env}_0  --data_dir=<data_path>  --algorithm MIRO --swad True --ld 0.0   --test_envs $test_env --dataset $dataset  --seed 0 --trial_seed 0 --model openclip_vit-b16
```

To train MIRO:

```
python  train_all.py -m  miro_${test_env}_0  --data_dir=<data_path>  --algorithm MIRO --swad False --ld 1.0   --test_envs $test_env --dataset $dataset  --seed 0 --trial_seed 0 --model openclip_vit-b16
```

To train Attn. Tune:

```
python   train_all.py -m  erm_attn_${test_env}_0  --data_dir=<data_path>  --algorithm MIRO --swad False --ld 0.0   --test_envs $test_env --dataset $dataset  --seed 0 --trial_seed 0 --model openclip_vit-b16 --attn_tune
```

To train MIRO + SWAD:

```
python train_all.py -m  miro_swad_${test_env}_0  --data_dir=<data_path>  --algorithm MIRO --swad True --ld 1.0   --test_envs $test_env --dataset $dataset  --seed 0 --trial_seed 0 --model openclip_vit-b16
```

To train MIRO + MPA:

```
python  train_all.py -m miro_mpa_lpft_${test_env}_0  --data_dir=<data_path>  --algorithm MIRO --swad True --ld 1.0   --test_envs $test_env --dataset $dataset  --seed 0 --trial_seed 0 --model openclip_vit-b16 --mpa
```

 To evaluate, change the dataset path to where Dataset-Easy/Hard is downloaded and add an --evaluate flag and --resume flag to where the model outputs are saved, eg.

 ```
 python   train_all.py -m  eval_erm_attn_${test_env}_0  --data_dir=<data_path>  --algorithm MIRO --swad False --ld 0.0   --test_envs $test_env --dataset $dataset  --seed 0 --trial_seed 0 --model openclip_vit-b16  --evaluate --resume_path train_output/${dataset}/latest_erm_attn_${test_env}_0/checkpoints/TE${test_env}_best.pth
 ```