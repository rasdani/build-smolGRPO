from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json

from if_functions import IF_FUNCTIONS_MAP

def if_reward(completions, ground_truth, **kwargs):
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        content = completion[0]["content"]
        gt = json.loads(gt)
        func_name = gt.pop("func_name")
        func = IF_FUNCTIONS_MAP[func_name]
        non_none_args = {k: v for k, v in gt.items() if v is not None}
        rewards.append(float(func(content, **non_none_args)))
    return rewards


model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("rasdani/smol-RLVR-IFeval", split="train")
train_dataset = dataset.select_columns(["messages", "ground_truth"]).rename_columns(
    {"messages": "prompt"}
)

num_gpus = 2
run_name = "ifeval_" + model_name.split("/")[-1].lower()

training_args = GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    learning_rate=1e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=20,
    num_train_epochs=1,
    bf16=True,
    adam_beta1=0.9,
    adam_beta2=0.99,
    max_grad_norm=0.1,
    num_iterations=1,
    beta=0.04,
    max_prompt_length=1024,
    max_completion_length=1024,
    # per_device_train_batch_size=4,
    per_device_train_batch_size=32,
    # per_device_train_batch_size=16,
    # num_generations=(2 * num_gpus - 2 if num_gpus > 1 else 2),
    num_generations=8,
    # gradient_accumulation_steps=int(16 / num_gpus),
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    save_strategy="steps",
    # save_steps=100,
    save_steps=200,
    save_only_model=True,
    use_vllm=True,
    vllm_device=f"cuda:{num_gpus-1}",
    vllm_gpu_memory_utilization=0.7 if num_gpus > 1 else 0.3,
    logging_steps=1,
    # log_on_each_node=False,
    log_on_each_node=True,
    log_completions=True,
    report_to="wandb",
    )

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=if_reward,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
)

trainer.train()
