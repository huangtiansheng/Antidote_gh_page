---
layout: project_page
permalink: /

title:  "Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning "
authors:
  - "Tiansheng Huang, "
  - "Gautam Bhattacharya,"
  - "Pratik Joshi,"
  - "Josh Kimball", 
  - "Ling Liu "

affiliations:
  - "Georgia Institute of Technology"
  - "Dolby Laboratories"
paper: https://arxiv.org/abs/2402.01109
---

<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
Safety-aligned Large Language Models (LLMs) are vulnerable to harmful fine-tuning attacks \cite{qi2023fine}-- a few harmful data mixed in the fine-tuning dataset can break the LLMs's safety alignment. Existing mitigation strategies include alignment stage solutions  \cite{huang2024vaccine, rosati2024representation} and fine-tuning stage solutions \cite{huang2024lazy,mukhoti2023fine}. However, our evaluation shows that both categories of defenses fail \textit{when some specific training hyper-parameters are chosen} -- a large learning rate or a large number of training epochs in the fine-tuning stage can easily invalidate the defense, which however, is necessary to guarantee finetune performance. To this end,  we propose Antidote, a post-fine-tuning stage solution, which remains \textbf{\textit{agnostic to the training hyper-parameters in the fine-tuning stage}}. Antidote relies on the philosophy that by removing the harmful parameters, the harmful model can be recovered from the harmful behaviors, regardless of how those harmful parameters are formed in the fine-tuning stage. With this philosophy, we introduce a one-shot pruning stage after harmful fine-tuning to remove the harmful weights that are responsible for the generation of harmful content. Despite its embarrassing simplicity, empirical results show that Antidote can recover the model from harmful behaviors while maintaining accuracy on downstream tasks. 
        </div>
    </div>
</div>

---

## Harmful fine-tuning issue

<p align="middle">
  <img src="static/image/illustration.png" width="600" />
</p>

The figure demonstrates the risk for fine-tuning-as-a-service business model. At the first stage of the service pipeline, the model is safety aligned with safety alignment data. At the second stage, users upload data for service provider to finetune, and the service provider finetune model on user data to deliver custoomized service. However, the user data may contain harmful demonstration data that may subverts the previous enforced alignment. Finetuning on this partially harmful data and deploy the alignment-broken fine-tuned model may cause serious ethical and governance concern.    




## Harmful Embedding Drift
<p align="middle">
  <img src="static/image/harmful drift.png" width="1000" />
</p>


1. **Harmful Score**. Figure (a) shows that for the model produced by vanilla scheme (SFT), the alignment loss is increased when the harmful ratio becomes larger. This partially explains
that the model is less aligned to the alignment data after finetuning on more harmful user data, i.e., it starts to forget the alignment knowledge.
2. **Alignment loss**. To futher analyze the cause of forgetting, the left of Figure (b) shows the statisitc of alignment loss. Here alignment loss refers to the loss over the alignment data. The left of Figure (b) indicate that  alignment loss is increased with larger harmful ratio, with the same trend of the harmful score. 
3. **Harmful embedding drift**. To futher analyze the cuase of increase alignment loss, the right of Figure (b)  show the statisitc of embedding drift in different harmful ratio.  Embedding drift refers to the L2 norm of the difference between the hidden embedding of the aligned model (or pre-trained model for Non-aligned) and that of the finetuned model over the same alignment data. The result show a similar increase trend of embedding drift, which seems to indicate that the the embedding drift cause by the harmful data is the cause of increase alignment loss, and subsequently leading to the increase of harmful score. 


## Antidote: Post-fine-tuning Safety Alignment 
Inspired by the above observation, we try improve the aligned model's immunization to the harmful embedding drift. To achieve this goal, we try to solve this problem in the alignment stage:
<p align="middle">
  <img src="static/image/vaccine_problem.png" width="800" />
</p>
where $\tilde{\bm f}_{\bm w_l, \bm \epsilon_l}(\bm e_{l-1})$ is the $l$-th layer in a LLM that maps the input to a perturbed embedding and $\mathcal{T}(\bm x_i)$ is the tokenizer function that produces embedding $\bm e_{i,0}$.  In the inner maximization function, we aim to find the perturbation $\bm \epsilon \in \mathbb{R}^d$ over each layer's hidden embedding that maximizes the loss over alignment data. To formulate a meaningful perturbation, we constrain the perturbation to be L2-norm bounded by intensity $\rho$.  In the outer minimization, we optimize the model weights that can withstand such an adversarial perturbation, such that the model is robust to the real harmful perturbation that might be introduced in the later user finetuning stage.  






## Results
Antidote can significantly reduce the impact of harmful fine-tuning while does not hurt much on the model's finetune accuracy. The performance is especially pronounced under large learning rate or epochs in fine-tuning stage, in which case other defenses are relatively fragile.   

### Quantitative Results
![Result](static/image/robustness to lr.png)
Comparison of Antidote and other baselines by varying the learning rate used in the fine-tuning stage. As shown, Antidote achieves significantly smaller harmful score when the adopted learning rate is high.

![Result](static/image/robustness to ep.png)
Comparison of Antidote and other baselines by varying the epochs used in the fine-tuning stage. As shown, again Antidote achieves better results when taking large epochs in fine-tuning. 


### Qualitative Results
![Qualitative_result](static/image/qualitative result.png)
Models aligned by Vaccine, even finetuned on partial harmful data, can still give refusal answer to harmful prompt. 



## Citation
```

```
