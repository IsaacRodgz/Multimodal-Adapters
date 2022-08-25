# Multimodal-Adapters

Adapter modules with support for multimodal fusion of information (text, video, audio, etc.) using pre-trained BERT base model. For a more detailed review of the architecture, refere to my [master thesis pdf](file:///Users/isaac/Downloads/TE%20832.pdf). in section 4.2 "Parameter-Efficient Transfer Learning for Multimodal Tasks" I describe the architectural changes made in order for the BERT model to support multimodal inputs.

A journal paper is on the way to present some interesting results with this architecture.

## Experiments

The proposed architecture was used to perform experiments with a movie genre multimodal classification task ([Moviescope](https://www.cs.virginia.edu/~pc9za/research/moviescope.html)). Multimodal-Adapter was compared with [MMBT](https://arxiv.org/abs/1909.02950), showing on par performance with a significant reduction in the number of parameters modified during finetuning. For more details about the experiments please refer section 5.5 "Multimodal Adapter Experiments" [here](file:///Users/isaac/Downloads/TE%20832.pdf).

## Experiments based on:

[Adapters](https://arxiv.org/abs/1902.00751) "Parameter-Efficient Transfer Learning for NLP" by Houlsby et al.

[Adapter Fusion](https://arxiv.org/abs/2005.00247) "AdapterFusion: Non-Destructive Task Composition for Transfer Learning" by Pfeiffer et al.

[Supervised Multimodal Bitransformers for Classifying Images and Text
](https://arxiv.org/abs/1909.02950)

[Moviescope: Large-scale Analysis of Movies using Multiple Modalities
](https://arxiv.org/abs/1908.03180)
