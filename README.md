# Is Grokking WorthWhile? Functional Analysis and Transferability of Generalization Circuits in Transformers

This repo contains several research codebases. Follow the following steps to reproduce our results in paper.
1) install packages,
2) generate synthetic data and train a traditional transformer,
3) preprocess data for TRM and run TRM pretrain/finetune,
4) visualize results.

## 1) Install

Recommended (Conda explicit environment):

```bash
cd  IsGrokkingWorthwhile
conda create -n explainableLLM_repro --file conda-explicit.txt
conda activate explainableLLM_repro
python -m pip install -r requirements.txt
```

## 2) Generate synthetic data + train baseline transformer

Generate the dataset by running the notebook:
- `IsGrokkingWorthwhile/Grokking_analysis/composition augmentation.ipynb`

The notebook saves json files under `IsGrokkingWorthwhile/Grokking_analysis/data/<dataset_name>/` (e.g., `train.json`, `valid.json`, `test.json`, `vocab.json`).

Then train the baseline transformer (command from the last markdown cell of the notebook):

Under Grokking_analysis folder
```bash
  PYTHONPATH="$PWD/simpletransformers:${PYTHONPATH}" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  python main.py \
    --data_dir data/composition.2000.200.18.0_factaug_h1ratio0.5_h1k9_h2ratio0.5_h2k9 \
    --model_name_or_path gpt2 \
    --weight_decay 0.01 \
    --output_dir output/composition.2000.200.18.0_factaug_h1ratio0.5_h1k9_h2ratio0.5_h2k9 \
    --max_seq_length 10 --max_length 10 --block_size 10 \
    --train_batch_size 512 --eval_batch_size 512 \
    --learning_rate 1e-4 --gradient_accumulation_steps 1 \
    --save_step 50000 --save_step_dense 40000 \
    --max_steps 1500000 \
    --do_train \
    --scheduler constant_schedule_with_warmup \
    --fp16 \
    --evaluate_during_training --predict_during_training \
    --init_weights --add_tokens \
    --n_layer 4 \
    --evaluate_train
```

Adjust `--data_dir`, `--output_dir`, and `CUDA_VISIBLE_DEVICES` as needed.

## 3) TRM preprocessing + pretrain/finetune

### 3.1 Preprocess

Run:
- `IsGrokkingWorthwhile/TRM_model/preprocess_training_data.ipynb`

This produces TRM-ready datasets under `IsGrokkingWorthwhile/TRM_model/data/<dataset_name>/`.

### 3.2 Pretrain

From `IsGrokkingWorthwhile/TRM_model/` (command copied from `preprocess_training_data.ipynb`):

```bash
run_name="pretrain_grok_composition" && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain_grok_evaluate_ver_0_1.py   arch=trm   data_paths="[data/composition.2000.200.18.0_factaug_h1ratio0.5_h1k9_h2ratio0.5_h2k9]"   evaluators="[]"   epochs=1100   eval_interval=5   lr=4e-5   puzzle_emb_lr=1e-4   weight_decay=1.0   puzzle_emb_weight_decay=1.0   arch.mlp_t=True   arch.pos_encodings=None   arch.L_layers=2   arch.H_cycles=2   arch.L_cycles=6 arch.halt_max_steps=1 arch.hidden_size=1536 +run_name=${run_name}  ema=True   global_batch_size=512  +max_inference_steps=1 checkpoint_every_eval=True   +format="maintain_prefix" +causal=False +post_fix="anything_here_you_like"
```

### 3.3 Finetune from a checkpoint (new dataset)

From `IsGrokkingWorthwhile/TRM_model/` (command copied from `preprocess_training_data.ipynb`):

```bash
run_name="TRM_finetune" && LOAD_CKPT={Your TRM checkpoint path here} && CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain_grok_evaluate_ver_0_1.py arch=trm data_paths="[{Your finetune datapath here}]" evaluators="[]" +load_checkpoint="${LOAD_CKPT}" epochs=20000 eval_interval=50 lr=2e-5 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 arch.mlp_t=True arch.pos_encodings=None arch.L_layers=2 arch.H_cycles=2 arch.L_cycles=6 arch.halt_max_steps=1 arch.hidden_size=1536 ema=True global_batch_size=512 +max_inference_steps=1 checkpoint_every_eval=False +causal=False +run_name="${run_name}" +post_fix="2000.200.18.0_no_aug_finetuning"
```

## 4) Visualize

- Pretrain and finetune analysis: `IsGrokkingWorthwhile/Grokking_analysis/Visualization Figures.ipynb`
