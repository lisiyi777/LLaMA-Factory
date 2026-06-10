# RF20-VL ShareGPT4V Data, Fine-Tuning, and Evaluation

This folder contains the scripts used to convert RF20-VL detection datasets into the ShareGPT4V conversation format used by LLaMA-Factory, then fine-tune Qwen-VL models and evaluate them.

## 1. Generate Prompts

`prompt_generation.py` reads the RF20-VL dataset README and COCO annotations, then creates one prompt file per dataset.

Inputs:

- `README.dataset.txt`: dataset description, class descriptions, and annotation instructions
- `train/_annotations.coco.json`: COCO categories used to match class names

Output:

- `{dataset_name}_prompts.json`

Each prompt file contains two prompt styles:

- `label_only`: a simple detection prompt using class names
- `with_description`: a detection prompt augmented with dataset-specific class descriptions and annotator instructions

Before running, update `root_dir` in `prompt_generation.py` to your local RF20-VL path:

```bash
python rf20vl_sharegpt4v/prompt_generation.py
```

## 2. Generate ShareGPT4V Datasets

`data_generation_sharegpt4v.py` converts COCO annotations and generated prompts into ShareGPT4V JSON files.

Inputs:

- `{dataset_name}_prompts.json` from the previous step
- `{split}/_annotations.coco.json` for `train`, `valid`, and `test`
- image files under each split directory

Output:

```text
sharegpt4v_datasets_refined/
  dataset-name/
    train/
      by_image_label_only.json
      by_image_with_description.json
      by_class_label_only.json
      by_class_with_description.json
    valid/
      ...
    test/
      ...
```

The four dataset variants are:

- `by_image_label_only`: multi-class prompt, class names only
- `by_image_with_description`: multi-class prompt with class descriptions/instructions
- `by_class_label_only`: single-class prompt, class name only
- `by_class_with_description`: single-class prompt with class description/instructions

For `valid` and `test`, the by-class files query every class for every image, so false positives can be measured during evaluation.

Before running, update `root_dir` and `output_base_dir` in `data_generation_sharegpt4v.py`:

```bash
python rf20vl_sharegpt4v/data_generation_sharegpt4v.py
```

## 3. Fine-Tune

```bash
python train_rf20vl.py
```

## 4. Evaluate

Example with LoRA adapters:

```bash
python rf20vl_sharegpt4v/evaluate_sharegpt4v_vllm.py \
  --eval_mode by_image_with_description \
  --sharegpt_root /path/to/sharegpt4v_datasets_refined \
  --data_dir /path/to/RF20VL_ROOT \
  --base_model_path Qwen/Qwen3-VL-8B-Instruct \
  --lora_model_path_root saves/qwen3vl-8b \
  --batch_size 4 \
  --save_dir results/rf20vl
```

Valid `--eval_mode` values are:

- `by_class_label_only`
- `by_class_with_description`
- `by_image_label_only`
- `by_image_with_description`

When `--lora_model_path_root` is provided, the evaluator expects adapters under:

```text
LORA_ROOT/
  dataset-name/
    single_class/
    single_instruction/
    multi_class/
    multi_instruction/
```

For example, `--eval_mode by_image_with_description` looks for `LORA_ROOT/dataset-name/multi_instruction`.

