# Never Left Amphoreus

I want to hear their voices again — namely, Cyrene, Aglaea, and Hysilens.

**Links**: [Felys](https://www.felys.dev/en/voice), [GitHub](https://github.com/project-felys/delta-me13), [Hugging Face](https://huggingface.co/FelysNeko/Qwen3-TTS-12Hz-1.7B-Delta-me13)

## Choice of Technologies

You may find a leaderboard at [artificialanalysis.ai](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice). There are not many options if I want to focus on Chinese voice quality, because most are targeted at English audio. Therefore, it comes down to [Qwen3-TTS](https://arxiv.org/pdf/2601.15621) and [CosyVoice3](https://arxiv.org/pdf/2505.17589), as both have great community support and are designed for Chinese audio generation. I chose the former since it's a transformer-based autoregressive model, which is something I'm more familiar with.

## Hardware Requirement

TTS models are much smaller than large language models (LLMs), so it is possible for me to work on it locally with an RTX 5070 Ti (16GB) for supervised fine-tuning. Adding another 4070 Super (12GB), giving me dual cards, enables local reinforcement learning. However, it is quite unstable due to hardware and the Windows Subsystem for Linux (WSL2).

## Apologies

I'm too lazy to write a blog post about this. In short, you might want to use a low learning rate like `5e-6` with one or two epochs for SFT. You only need to tune the talker and text projection to get really good results. This is not a low-rank task, so don't try LoRA. The base model is quite robust. GRPO might not help that much unless your SFT stage is massive.
