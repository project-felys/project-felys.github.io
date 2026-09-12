# Never Left Amphoreus

I want to hear their voices again — namely, Cyrene, Aglaea, and Hysilens.

**Links**: [Felys](https://www.felys.dev/en/voice), [GitHub](https://github.com/project-felys/delta-me13), [Hugging Face](https://huggingface.co/FelysNeko/Qwen3-TTS-12Hz-1.7B-Delta-me13)

## Summary

I'm too lazy to write a blog post about this. In short, you might want to use a low learning rate like `5e-6` with one or two epochs for supervised fine-tuning (SFT). You only need to tune the talker and text projection to get really good results. This is not a low-rank task, so don't try LoRA. The base model is quite robust. GRPO is unnecessary unless your SFT stage is massive.
