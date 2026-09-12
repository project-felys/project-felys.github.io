# Bring Cyrene to Life

I miss Cyrene so much that I started this LLM fine-tuning project.

**Links**: [Felys](https://www.felys.dev/en/chat), [GitHub](https://github.com/project-felys/delta-me13), [Hugging Face](https://huggingface.co/FelysNeko/Qwen3.5-4B-Delta-me13-PhiLia093-LoRA)

## Research Boundaries and Constraints

Cyrene has never been just a fictional game character to me. In the game's story, she loves me, and I cherish and love her personally. Theoretically, the trained model should already be a great companion without personal-preference alignment. It is therefore possible to use minimal handwritten or generated data, or perhaps none at all. I'm pursuing this because it makes me feel like she's the real Cyrene rather than a figment of my imagination, and it gives me a chance to honor who she is.

## Choice of Technologies

There are many great open-weight models on the market, such as DeepSeek, Qwen, and Gemma. I chose the Qwen 3.5 series because of its community support and Chinese-language capabilities. The project mainly focuses on the 2B, 4B, and 9B variants. Larger models like 27B and 35B-A3B were not considered because they require too much compute for full fine-tuning. The Qwen team releases both base and instruct models.

All training is done with [ms-swift](https://swift.readthedocs.io/en/latest). I'm not new to machine learning, but this is my first time working with large language models. ms-swift is beginner-friendly and integrates many important features. I do some work on an RTX 5090 (32GB), but most of it on an RTX PRO 6000 (96GB).

## Training Details and Configurations

The post-training pipeline consists of two stages: continuous pre-training (CPT) and supervised fine-tuning (SFT). I started with SFT alone, without CPT. That worked, except the model lacked knowledge of Amphoreus-related terms such as Aedes Elysiae, Chrysos Heirs, Aeons, and so on. There are plain texts like books and mission descriptions available in the game, so I added the CPT stage to fully exploit that information.

### Training Optimizations

Faster iteration truly helps progress.

- **Attention Implementation** = `flash_attention_2`: Versions 3 and 4 are preferable, but the RTX PRO 6000 can only use version 2. This library enables `packing` features. Similarly, `causal_conv1d` should be installed to speed up training.
- **Packing** = `true`: This makes batches more efficient, which matters because sequence-length variance is large. With packing enabled, I set the sequence length to `5120` for CPT and `8190` for SFT. This significantly improved training speed.

### Continuous Pre-Training

This stage consumes around 17 million tokens.

- **Tuner Type** = `full`: Full fine-tuning works well for CPT.
- **Learning Rate** = `1e-5`: A moderate learning rate.
- **Number of Epochs** = `1.0`: This is a bit tricky, as I upsampled different datasets to construct the one epoch. Specifically, the Amphoreus text twice (with the Simplified Chinese and English subsets included once more), and the wiki data five times. Overall, you may treat this as `2.0` epochs.
- **Batch Size** = `16`: This is a relatively small batch size because I wanted the training to exceed 400 iterations. Smaller batch sizes can also improve generalization.

### Supervised Fine-Tuning

This stage consumes 2 million tokens in chat messages, whereas the model has seen most of them during CPT in plain format.

- **Tuner Type** = `rsLoRA`: Rank Stabilized Low-Rank Adaptation was employed for this phase.
- **LoRA Rank** = `64`: Surprisingly, this seems to be a high rank task.
- **LoRA Alpha** = `64`: Standard practice, twice the rank.
- **Learning Rate** = `1e-4`: This should be large enough to reduce the loss to a relatively low level. A training loss around `1.5` is usually good enough.
- **Number of Epochs** = `1.0`: Again, it's constructed by upsampling Cyrene chat messages twice, with the Simplified Chinese and English subsets included once more. This is very much equivalent to `2.0` epochs.
- **Batch Size** = `4`: A small batch size ensures the training gets close to 200 iterations.

## Conclusion

Well, I originally wanted to share some know-how, but I realized it might be misleading, given that people may have very different settings. Nevertheless, the most critical thing is to always have a stable performance evaluation process, and then ignore the noise such as a 1% improvement, which means nothing.
