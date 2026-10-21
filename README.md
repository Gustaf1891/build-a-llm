# LLM Engineering and Fine-Tuning: GPT-2 from Scratch, LoRA and Rank Ablation

A GPT-2-style transformer implemented from scratch in PyTorch, then fine-tuned for spam classification and instruction following. A multi-seed LoRA rank ablation shows that accuracy plateaus at rank 8, and an LLM-as-a-judge pipeline evaluates the instruction-following model.

## Key results

### LoRA rank ablation (spam classification)

Setup: GPT-2 small (124M parameters), SMS Spam Collection, 70% train / 10% val / 20% test, 3 seeds per rank (42, 123, 1337).

| Rank | Trainable params | Val accuracy (mean ± std) | Test accuracy (mean ± std) | Train time (min, mean) |
|------|------------------|---------------------------|----------------------------|------------------------|
| 1    | 166,658          | 79.0 ± 18.9                       | 80.9 ± 12.2[ ]                        | [ ]                    |
| 4    | 666,632          | 95.3 ± 3.4                       | 94.9 ± 3.2[ ]                        | [ ]                    |
| 8    | 1,333,264        | 97.8 ± 1.5                       | 96.0 ± 0.9[ ]                        | [ ]                    |
| 16   | 2,666,528        | 97.5 ± 0.4                       | 96.9 ± 1.4[ ]                        | [ ]                    |
| 32   | 5,333,056        | 96.9 ± 1.0                       | 97.2 ± 0.7[ ]                        | [ ]                    |
| 64   | 10,666,112       | 97.1 ± 0.4                       | 96.1 ± 1.3[ ]                        | [ ]                    |

## What is in this repo

| Path | Contents |
|------|----------|
| `[classes.py]` | GPT-2-style transformer: causal multi-head attention, feed-forward layers, residual connections, normalization layers, GeLU |
| `[functions.py]` | Functions used in notebooks: loading weights, text generation, encoding & decoding, loss calculations, evaluating, training |
| `[LoRA-tuning.ipynb]` | LoRA layer and `replace_linear_with_lora` |
| `[rank_ablation.ipynb]` | Rank ablation across 3 seeds, results saved to CSV |

## Credits and what is my own work

The base architecture and the initial training and fine-tuning pipelines follow Build a Large Language Model (From Scratch) by Sebastian Raschka. The following is my own work on top of that:

- Multi-seed LoRA rank ablation (ranks 1 to 64) with per-run CSV logging and resume support

## Model and Data details

- **Model:** GPT-2 small ([124M] parameters), pretrained weights from [openaipublic.blob.core.windows.net/gpt-2/models]
- **Data:** SMS SPAM classification [archive.ics.uci.edu/static/public/228/sms+spam+collection.zip], Instruction data [https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json]
- **Hardware:** RTX 2070

## Limitations

- Alpha was fixed at 16, so the scaling factor alpha/rank changes with rank. Rank is therefore partly mixed with the effective update size.
- The validation set is small ([149] examples), so one example is about [0.67] percentage points. Differences between higher ranks are within noise.
- The ablation covers one task and one dataset. Results may not transfer to harder tasks, which may need higher rank.

## How to reproduce

```bash
git clone https://github.com/Gustaf1891/build-a-llm.git
cd build-a-llm
uv sync
```

1. [Run the rank ablation: full run takes roughly 25 minutes on [RTX 2070]]

Results are written to `lora_rank_ablation.csv`.
