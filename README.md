# build-smolGRPO


```
trl vllm-serve --model Qwen/Qwen2.5-0.5B
```



```
accelerate launch --config_file configs/deepspeed/zero3.yaml --num_processes 1 simple_train.py
```