# Multi-Language Round-Trip Translation Evaluation

This project benchmarks how well different LLMs preserve meaning under **round-trip translation** (English → X → English) across several target languages, and explores whether **LoRA fine-tuning** can close the gap between small open models and a frontier API model on English↔Mandarin Chinese.

Source sentences come from the [`syvai/finetranslations-chat-da`](https://huggingface.co/datasets/syvai/finetranslations-chat-da) dataset. Translation quality is scored with **BLEU**, **chrF**, **BERTScore F1**, and (in the final evaluation) **BLEURT**.

## Models compared

- **Claude Haiku 4.5** — via the Anthropic API
- **Qwen3-4B-Instruct-2507** — base model, run locally in 4-bit on a T4 GPU
- **Qwen3-4B + LoRA (Tatoeba)** — fine-tuned for EN↔ZH on Tatoeba sentence pairs
- (Plus Llama-3.2-3B variants generated in a separate sister project, included in the final consolidated evaluation)

## Files

### `AI_translation__1_.ipynb`
The Claude Haiku translation pipeline. Streams 500 source sentences from the dataset, round-trip translates them through Spanish, Korean, Chinese, Swahili, and Maltese via the Anthropic API, and scores results with BLEU / chrF / BERTScore. Includes a custom rate limiter that respects Tier 1 RPM + OTPM limits, per-language CSV checkpoints for crash recovery, and a 5-panel visualization comparing quality across languages.

### `AI_translation_qwen.ipynb`
Same round-trip pipeline as above, but using **Qwen3-4B-Instruct-2507** running locally on a Colab T4 GPU in 4-bit quantization. Adds German as a sixth language and uses batched GPU inference instead of API calls. Outputs are saved to Google Drive (`lang_results_qwen/`) for downstream evaluation.

### `train_qwen.ipynb`
Fine-tunes Qwen3-4B-Instruct-2507 on English↔Mandarin Chinese translation using **Unsloth + QLoRA** on a single T4. Trains two separate LoRA adapters — one for EN→ZH and one for ZH→EN — on ~32k Tatoeba sentence pairs. Uses ChatML formatting, cosine LR schedule, eval-loss-based best-checkpoint selection, and rsyncs checkpoints to Google Drive.

### `eval_finetuned_qwen.ipynb`
Evaluates the fine-tuned Qwen3-4B LoRA adapters on round-trip EN→ZH→EN translation, comparing them directly against the base Qwen3-4B baseline from `AI_translation_qwen.ipynb`. Loads each adapter in turn, runs batched inference, and saves results for the consolidated evaluation.

### `evaluate_all.ipynb`
The final consolidated evaluation. Loads the round-trip results from all six model variants (Claude Haiku, Llama-3.2-3B base, Llama-3.2-3B-FT-WMT, Llama-3.2-3B-FT-Tatoeba, Qwen3-4B base, Qwen3-4B-FT-Tatoeba) and scores them with BLEU, chrF, BERTScore F1, and BLEURT. Produces side-by-side comparisons across languages and models — the answer to "did fine-tuning help, and how close does it get to Claude?"

## Typical workflow

1. Run `AI_translation__1_.ipynb` to generate the Claude baseline.
2. Run `AI_translation_qwen.ipynb` to generate the base-Qwen baseline.
3. Run `train_qwen.ipynb` to produce the two LoRA adapters.
4. Run `eval_finetuned_qwen.ipynb` to generate fine-tuned Qwen translations.
5. Run `evaluate_all.ipynb` to compare everything in one place.

## Requirements

- Anthropic API key (for the Claude pipeline)
- Google Colab with T4 GPU (for the Qwen pipelines)
- Google Drive mounted (for cross-notebook output sharing)
