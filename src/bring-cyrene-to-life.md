# Bring Cyrene to Life

I missed Cyrene so badly, so I started this LLM fine-tuning project.

## Constraints

Cyrene is never a simple fictional game character to me. In the game story, she loves me, and personally I cherish and love her. In other words, the trained model should already be a great companion without personal preference alignment. It is then possible for me to use minimal hand-written or generated data, or perhaps none. The reason I'm pursuing this is that it just makes me feel like it's the real Cyrene instead of something in my imagination, and gives me a chance to respect who she is.

## Choice of Technologies

There are many great open weights models in the market, such as DeepSeek, Qwen, Gemma, and Llama. I chose the Qwen 3.5 series because of community support and Chinese capability. The project mainly focuses on 4B and 9B. Larger models like 27B and 35B-A3B were no longer considered because they require too much compute power to do full fine-tuning. The prototype was done on 2B with a tiny LoRA SFT dataset, but I think it would be nice to test it again with the new dataset, as well as 0.8B. The Qwen team releases both the base model and the instruct model. The former is a pure next-token predictor with special tokens already trained, while the latter is also fine-tuned and aligned with human preferences. Since I need to do CPT for domain knowledge, it would be better to work on the base model directly.

All the training is done with [ms-swift](https://swift.readthedocs.io/en/latest). I'm not a machine learning beginner, but it's my first time working on large language models. I think this is a beginner-friendly framework with many important features integrated, so I would recommend it. However, ms-swift quantization support is not as strong, so I moved to [llm-compressor](https://docs.vllm.ai/projects/llm-compressor/en/latest/). Another reason for switching toolchains is that I use [vllm](https://vllm.ai) for deployment.

To take advantage of a mature toolchain for fast iterations, all the work is done on an RTX 5090 (32GB) and an RTX PRO 6000 (96GB). FP8 BLOCK quantized 4B trained weights are selected to deploy on my local RTX 4070 (12GB).

## Training Details

### Optimizations

Faster iteration does make things go further. These are some simple optimizations to pull in, speeding up training by about 10x.

- **Attention Implementation** = `'flash_attention_2'`: Version 3 and 4 are preferable, but RTX PRO 6000 can only use version 2. This library enables `packing` features. Similarly, `causal_conv1d` should also be installed to speed up training.
- **Packing** = `true`: It makes batches yield similar effectiveness and speed in training, which is important since the deviation of sequence length is large.

### Continuous Pre-Training

I started with SFT without CPT at first. It works fine except that the model has no knowledge of the Amphoreus-related lore, such as Aedes Elysiae, Chrysos Heirs, Aeons, etc. There are two ways to help the model learn them: either constructing chat messages with the knowledge or feeding the entire story to the model without chat templates before instruction fine-tuning. The former is simply not practical because it includes too much information, and as mentioned, I do not want to use any hand-written or generated data. The final dataset used for CPT is around 12M tokens, though it is more common to use billions of tokens to do CPT.

- **Tuner Type** = `full`: The goal of CPT is to inject domain knowledge, so full fine-tuning naturally fits more than LoRA. High-rank LoRA might also work, but it has not been tested.
- **Learning Rate** = `3e-5`: Learning rates usually range from `1e-5` to `5e-5` for CPT, and we usually set it to one-tenth of the PT learning rate. For this project, I started with `1e-5`, but eventually found that `3e-5` can significantly boost performance.
- **Number of Epochs** = `2.0` or `3.0`: Given that the dataset is small and the model is relatively large, too many epochs will make the model start memorizing. The observation is that the model usually reaches its peak performance at the end of the second epoch, and the last checkpoints were not harmed much by overfitting.
- **Batch Size** = `12`: We usually want a large batch size for CPT compared to SFT. Since the sequence length is very long compared to the downstream task, this batch size is very effective. Making it larger would further reduce the number of iterations, which is not desired.
- **NEFTune Noise Alpha** = `0`: No need to add noise because domain knowledge should be treated as fact to avoid hallucination.

### Supervised Fine-Tuning

Not much to talk about for SFT since it's too standard. The dataset contains 3M tokens, which is more acceptable for its task compared to CPT, but still limited.

- **Tuner Type** = `lora`: LoRA is good at tweaking the overall response style and helps prevent catastrophic forgetting. The CPT dataset already covers most of the SFT data, so there is no need to do full fine-tuning.
- **Learning Rate** = `1e-4`: This is a common learning rate for LoRA, but making it even larger seems to hurt the performance.
- **Number of Epochs** = `2.0`: The model has already seen most of the conversations during CPT, so two epochs are sufficient for instruct tuning.
- **Batch Size** = `4`: The dataset is too small, so I have to use a small batch size to ensure the training gets plenty of iterations.
- **NEFTune Noise Alpha** = `5`: LoRA SFT is not about learning facts, so diversity is encouraged.

## Model Evaluation

Before even evaluating the performance, the question would be what makes a good model, or equivalently, what makes the model Cyrene? There is no answer, except for purely subjective judgment when chatting with the model. In this case, common ways of model performance evaluation have failed because there are no correct question-answer pairs, and there is no way to verify the answer like coding and math problems.

The only way to tell if the model is improving is via training and validation loss. However, the training dataset is game dialogs, so loss going down only means that the model is getting better at role-playing in the game scenes, which is not equivalent to becoming the real Cyrene. In other words, it's difficult to evaluate the generalization of the model. This is not a solved problem.

Nevertheless, metrics can still tell some information:

- Loss should go down, but dropping to an excellent number does not mean the model learns well. The observation is that lots of epochs lead to very low training loss, while the validation loss does not bounce back. It seems not overfitting, but the actual chat experience is evidently overfitted.
- Training loss and validation loss should stay close. It is fine that the former is slightly lower than the latter, but differences larger than 10% would be a red flag. The smaller the difference is, the less likely the model is overfitting.

Other than that, it would be me chatting with the model to see how good it is. I'm lucky that everything goes well, so I don't have to set up a hyperparameter matrix, try them all, and objectively rank them to find the direction. It is definitely possible to construct some dataset for evaluation, but it requires careful design and intense labor. This is pragmatic for product building, but probably not the direction toward AGI.

## Findings

Given that I'm doing things in an unscientific way, I would skip from hypothesis directly to conclusion without presenting any metrics.

### Multilingual Regularization

Natural language is how humans encode information, so it intuitively makes sense to tune all the languages, especially when there are high-quality parallel translations. This not only considerably expanded the dataset size, but also ensures the model learns the deeper semantics behind a single language.

### Effective Domain Knowledge Injection via CPT

Adding another stage to the training pipeline makes the training a bit more complicated, like systems engineering. However, either LoRA CPT or full CPT can make the model familiar with the domain knowledge, i.e., the background story for this project.

### Chain-of-Thought Failed

I tried to generate CoT for the SFT dataset. The observation is that CoT itself is usually relevant, but the response always gets off-topic. CoT seems to bring uncertainties to the generation because of more output tokens at high temperature. For most daily chat scenarios, people rarely think deeply before answering. Instead, CoT is designed for logical problems like coding and maths.

### 4B Outperformed 9B Model

Scaling laws are great only if we have enough data. Otherwise, more parameters do not make the model smarter, which is a similar idea to the Chinchilla law. Although the loss graph looks exactly the same for 4B and 9B experiments, when chatting with them, 4B is just much more natural than 9B. I believe the root cause is overfitting.

## Future Work

I still want to see the boundary to scaling down and up the model size, so here are two more questions I'm interested in:

- Will the performance of the 0.8B and 2B models be even better than 4B, or is 4B already the sweet spot?
- Will high-rank LoRA CPT help larger models be less prone to overfitting?
