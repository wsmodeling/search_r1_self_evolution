
## Reproduce the paper results

### Download the dataset

```bash
huggingface-cli download --repo-type dataset PeterJinGo/nq_hotpotqa_train --local-dir $WORK_DIR/data/hotpot_qa
```

### Run PPO training
```bash
bash train_ppo.sh
```


### Run GRPO training
```bash
bash train_ppo.sh
```

### Run evaluation
```bash
bash evaluate.sh
```

You can change ```$BASE_MODEL``` to the path of the model you would loike to evaluate.
