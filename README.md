# Awesome-LLM-Eval

Awesome-LLM-Eval: a curated list of tools, demos, papers, docs for Evaluation on Large Language Models (like ChatGPT, LLaMA, GLM, etc).

## Tools

| 机构 | 名称 | 网址 |
| :--: | :--: | :--: |
| OPENAI | EVAL | https://github.com/openai/evals |
| EleutherAI | lm-evaluation-harness | https://github.com/EleutherAI/lm-evaluation-harness |
| wgryc | Large language model evaluation and workflow framework from Phase AI | https://github.com/wgryc/phasellm |
| FreedomIntelligence | evaluation benchmark for large language models | https://github.com/FreedomIntelligence/LLMZoo |
| Stanford | Holistic Evaluation of Language Models (HELM) | https://github.com/stanford-crfm/helm |
| Langchain | a lightweight evaluation tool for question-answering | https://github.com/rlancemartin/auto-evaluator |
| WeOpenML | PandaLMReProducible and Automated Language Model Assessment | https://github.com/WeOpenML/PandaLM |

## Demos
- [Chat with two anonymous models side-by-side and vote for which one is better](https://chat.lmsys.org/?arena) - 开源AI大模型“匿名”竞技场！你在这里可以成为一名裁判，给两个事先不知道名字的模型回答打分，评分后将给出他们的真实身份。目前已经“参赛”的选手包括Vicuna、Koala、OpenAssistant (oasst)、Dolly、ChatGLM、StableLM、Alpaca、LLaMA

## Datasets

- [JioNLP-LLM评测数据集](https://github.com/dongrixinyu/JioNLP/wiki/LLM评测数据集) - jionlp LLM 评测数据集主要用于评测通用 LLM 的效果评价。着眼考察 LLM 模型对人类用户的帮助效果、辅助能力，可否达到一个【智能助手】的水平。题型包括：选择题来源于中国大陆国内各种专业性考试，重点在于考察模型对客观知识的覆盖面，占比 32%；主观题来源于日常总结，主要考察用户对 LLM 常用功能的效果。
- [BIG-bench](https://github.com/google/BIG-bench) - google推出的一个评测集，BIG bench由 204 项任务组成，任务主题涉及语言学、儿童发展、数学、常识推理、生物学、物理学、社会偏见、软件开发等等领域的问题。
- [SuperCLUE](https://github.com/CLUEbenchmark/SuperCLUE) - 中文的一个榜单，这里从基础能力、专业能力、中文特性三个角度进行准备测试集 基础能力能力包括：语义理解、对话、逻辑推理、角色模拟、代码、生成与创作等10项能力。专业能力包括：包括了中学、大学与专业考试，涵盖了从数学、物理、地理到社会科学等50多项能力。中文特性能力：针对有中文特点的任务，包括了中文成语、诗歌、文学、字形等10项多种能力。
- [Safety Eval: Tsinghua University 清华安全大模型评测](https://github.com/thu-coai/Safety-Prompts) - 清华收集的一个评测集，涵盖了仇恨言论、偏见歧视言论、犯罪违法、隐私、伦理道德等八大类别，包括细粒度划分的40余个二级安全类别，并依托于一套系统的安全评测框架。网址：http://115.182.62.166:18000。
- [GAOKAO-Bench](https://github.com/OpenLMLab/GAOKAO-Bench) - GAOKAO-bench是一个以中国高考题目为数据集，测评大模型语言理解能力、逻辑推理能力的测评框架.
- [MMLU](https://paperswithcode.com/dataset/mmlu) - 该测评数据集涵盖 STEM、人文学科、社会科学等领域的 57 个学科。难度从初级到专业高级，既考验世界知识，又考验解决问题的能力。学科范围从数学和历史等传统领域到法律和伦理等更专业的领域。主题的粒度和广度使基准成为识别模型盲点的理想选择。
- [MMCU](https://paperswithcode.com/dataset/mmlu) - 甲骨易AI研究院提出一种衡量中文大模型处理多任务准确度的测试, 数据集的测试内容涵盖四大领域：医疗、法律、心理学和教育。题目的数量达到1万+，其中包括医疗领域2819道题，法律领域3695道题，心理学领域2001道，教育领域3331道。 
- [c_eval](https://arxiv.org/pdf/2305.08322.pdf) - 上交、清华以及爱丁堡大学合作产出的一个评测集，包含52个学科来评估大模型高级知识和推理能力，其评估了包含 GPT-4、ChatGPT、Claude、LLaMA、Moss 等多个模型的性能.

## Papers

- [<img src=https://img.shields.io/badge/CoRR-2023-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://www.microsoft.com/en-us/research/publication/gpteval-nlg-evaluation-using-gpt-4-with-better-human-alignment/) [**G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment**](https://www.microsoft.com/en-us/research/publication/gpteval-nlg-evaluation-using-gpt-4-with-better-human-alignment/),<br> by *Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, Chenguang Zhu*
- [<img src=https://img.shields.io/badge/CoRR-2023-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://doi.org/10.48550/arXiv.2302.04023) [**A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning,
Hallucination, and Interactivity**](https://doi.org/10.48550/arXiv.2302.04023),<br> by *Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji et al.*
<br><br>
- [<img src=https://img.shields.io/badge/CoRR-2023-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://arxiv.org/abs/2302.06476) [**Is ChatGPT a General-Purpose Natural Language Processing Task Solver?**](https://arxiv.org/abs/2302.06476),<br> by *Qin, Chengwei, Zhang, Aston, Zhang, Zhuosheng, Chen, Jiaao, Yasunaga, Michihiro and Yang, Diyi*
<br><br>
- [<img src=https://img.shields.io/badge/CoRR-2023-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://doi.org/10.48550/arXiv.2302.06466) [**ChatGPT versus Traditional Question Answering for Knowledge Graphs:
Current Status and Future Directions Towards Knowledge Graph Chatbots**](https://doi.org/10.48550/arXiv.2302.06466),<br> by *Reham Omar, Omij Mangukiya, Panos Kalnis and Essam Mansour*
<br><br>
- [<img src=https://img.shields.io/badge/CoRR-2023-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://doi.org/10.48550/arXiv.2301.13867) [**Mathematical Capabilities of ChatGPT**](https://doi.org/10.48550/arXiv.2301.13867),<br> by *Simon Frieder, Luca Pinchetti, Ryan-Rhys Griffiths, Tommaso Salvatori, Thomas Lukasiewicz, Philipp Christian Petersen, Alexis Chevalier and Julius Berner*
<br><br>
- [<img src=https://img.shields.io/badge/CoRR-2023-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://doi.org/10.48550/arXiv.2302.08081) [**Exploring the Limits of ChatGPT for Query or Aspect-based Text Summarization**](https://doi.org/10.48550/arXiv.2302.08081),<br> by *Xianjun Yang, Yan Li, Xinlu Zhang, Haifeng Chen and Wei Cheng*
<br><br>
- [<img src=https://img.shields.io/badge/CoRR-2023-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://doi.org/10.48550/arXiv.2302.12095) [**On the Robustness of ChatGPT: An Adversarial and Out-of-distribution
Perspective**](https://doi.org/10.48550/arXiv.2302.12095),<br> by *Jindong Wang, Xixu Hu, Wenxin Hou, Hao Chen, Runkai Zheng, Yidong Wang, Linyi Yang, Haojun Huang et al.*
<br><br>
- [<img src=https://img.shields.io/badge/CoRR-2023-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://doi.org/10.48550/arXiv.2301.04655) [**ChatGPT is not all you need. A State of the Art Review of large
Generative AI models**](https://doi.org/10.48550/arXiv.2301.04655),<br> by *Roberto Gozalo-Brizuela and Eduardo C. Garrido-Merch\'an*
<br><br>
- [<img src=https://img.shields.io/badge/CoRR-2023-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://arxiv.org/abs/2302.10198) [**Can ChatGPT Understand Too? A Comparative Study on ChatGPT and Fine-tuned
BERT**](https://arxiv.org/abs/2302.10198),<br> by *Qihuang Zhong, Liang Ding, Juhua Liu, Bo Du and Dacheng Tao*
<br><br>
- [<img src=https://img.shields.io/badge/CoRR-2023-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://doi.org/10.48550/arXiv.2303.07992) [**Evaluation of ChatGPT as a Question Answering System for Answering
Complex Questions**](https://doi.org/10.48550/arXiv.2303.07992),<br> by *Yiming Tan, Dehai Min, Yu Li, Wenbo Li, Nan Hu, Yongrui Chen and Guilin Qi*
<br><br>
- [<img src=https://img.shields.io/badge/CoRR-2023-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://arxiv.org/abs/2303.16421) [**ChatGPT is a Knowledgeable but Inexperienced Solver: An Investigation of Commonsense Problem in Large Language Models**](https://arxiv.org/abs/2303.16421),<br> by *Ning Bian, Xianpei Han, Le Sun, Hongyu Lin, Yaojie Lu and Ben He*
<br><br>
- [<img src=https://img.shields.io/badge/CoRR-2022-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://doi.org/10.48550/arXiv.2211.09110) [**Holistic Evaluation of Language Models**](https://doi.org/10.48550/arXiv.2211.09110),<br> by *Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan et al.*
<br><br>
- [<img src=https://img.shields.io/badge/CoRR-2022-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://doi.org/10.48550/arXiv.2204.00498) [**Evaluating the Text-to-SQL Capabilities of Large Language Models**](https://doi.org/10.48550/arXiv.2204.00498),<br> by *Nitarshan Rajkumar, Raymond Li and Dzmitry Bahdanau*
<br><br>
- [<img src=https://img.shields.io/badge/COLING-2022-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://aclanthology.org/2022.coling-1.491) [**Are Visual-Linguistic Models Commonsense Knowledge Bases?**](https://aclanthology.org/2022.coling-1.491),<br> by *Hsiu-Yu Yang and Carina Silberer*
<br><br>
- [<img src=https://img.shields.io/badge/CoRR-2022-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://doi.org/10.48550/arXiv.2212.10529) [**Is GPT-3 a Psychopath? Evaluating Large Language Models from a Psychological
Perspective**](https://doi.org/10.48550/arXiv.2212.10529),<br> by *Xingxuan Li, Yutong Li, Linlin Liu, Lidong Bing and Shafiq R. Joty*
<br><br>
- [<img src=https://img.shields.io/badge/EMNLP-2022-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://aclanthology.org/2022.emnlp-main.132) [**GeoMLAMA: Geo-Diverse Commonsense Probing on Multilingual Pre-Trained
Language Models**](https://aclanthology.org/2022.emnlp-main.132),<br> by *Da Yin, Hritik Bansal, Masoud Monajatipoor, Liunian Harold Li and Kai-Wei Chang*
<br><br>
- [<img src=https://img.shields.io/badge/EMNLP-2022-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://aclanthology.org/2022.emnlp-main.653) [**RobustLR: A Diagnostic Benchmark for Evaluating Logical Robustness
of Deductive Reasoners**](https://aclanthology.org/2022.emnlp-main.653),<br> by *Soumya Sanyal, Zeyi Liao and Xiang Ren*
<br><br>
- [<img src=https://img.shields.io/badge/CoRR-2022-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://arxiv.org/abs/2202.13169) [**A Systematic Evaluation of Large Language Models of Code**](https://arxiv.org/abs/2202.13169),<br> by *Frank F. Xu, Uri Alon, Graham Neubig and Vincent J. Hellendoorn*
<br><br>
- [<img src=https://img.shields.io/badge/CoRR-2021-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://arxiv.org/abs/2107.03374) [**Evaluating Large Language Models Trained on Code**](https://arxiv.org/abs/2107.03374),<br> by *Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pond\'e de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda et al.*
<br><br>
- [<img src=https://img.shields.io/badge/ACL-2021-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://doi.org/10.18653/v1/2021.findings-acl.36) [**GLGE: A New General Language Generation Evaluation Benchmark**](https://doi.org/10.18653/v1/2021.findings-acl.36),<br> by *Dayiheng Liu, Yu Yan, Yeyun Gong, Weizhen Qi, Hang Zhang, Jian Jiao, Weizhu Chen, Jie Fu et al.*
<br><br>
- [<img src=https://img.shields.io/badge/CoRR-2021-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://arxiv.org/abs/2104.05861) [**Evaluating Pre-Trained Models for User Feedback Analysis in Software
Engineering: A Study on Classification of App-Reviews**](https://arxiv.org/abs/2104.05861),<br> by *Mohammad Abdul Hadi and Fatemeh H. Fard*
<br><br>
- [<img src=https://img.shields.io/badge/ACL_Findings-2021-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://doi.org/10.18653/v1/2021.findings-acl.322) [**Do Language Models Perform Generalizable Commonsense Inference?**](https://doi.org/10.18653/v1/2021.findings-acl.322), [<img src=https://img.shields.io/badge/Code-skyblue alt="img" style="zoom:100%; vertical-align: middle" />](https://github.com/wangpf3/LM-for-CommonsenseInference)<br> by *Peifeng Wang, Filip Ilievski, Muhao Chen and Xiang Ren*
<br><br>
- [<img src=https://img.shields.io/badge/EMNLP-2021-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://doi.org/10.18653/v1/2021.emnlp-main.598) [**RICA: Evaluating Robust Inference Capabilities Based on Commonsense
Axioms**](https://doi.org/10.18653/v1/2021.emnlp-main.598),<br> by *Pei Zhou, Rahul Khanna, Seyeon Lee, Bill Yuchen Lin, Daniel Ho, Jay Pujara and Xiang Ren*
<br><br>
- [<img src=https://img.shields.io/badge/CoRR-2020-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://arxiv.org/abs/2006.14799) [**Evaluation of Text Generation: A Survey**](https://arxiv.org/abs/2006.14799),<br> by *Asli Celikyilmaz, Elizabeth Clark and Jianfeng Gao*
<br><br>
- [<img src=https://img.shields.io/badge/CoRR-2020-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://arxiv.org/abs/2007.15780) [**Neural Language Generation: Formulation, Methods, and Evaluation**](https://arxiv.org/abs/2007.15780),<br> by *Cristina Garbacea and Qiaozhu Mei*
<br><br>
- [<img src=https://img.shields.io/badge/ICLR-2020-blue alt="img" style="zoom:100%; vertical-align: middle" />](https://openreview.net/forum?id=SkeHuCVFDr) [**BERTScore: Evaluating Text Generation with BERT**](https://openreview.net/forum?id=SkeHuCVFDr),<br> by *Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger and Yoav Artzi*
<br><br>


## LLM List

### Pre-trained LLM

|       Model       | Size |  Architecture  |                                                                                               Access                                                                                               |  Date  | Origin                                                                                                                        |
| :----------------: | :--: | :-------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----: | ----------------------------------------------------------------------------------------------------------------------------- |
| Switch Transformer | 1.6T |  Decoder(MOE)  |                                                                                                   -                                                                                                   | 2021-01 | [Paper](https://arxiv.org/pdf/2101.03961.pdf)                                                                                    |
|        GLaM        | 1.2T |  Decoder(MOE)  |                                                                                                   -                                                                                                   | 2021-12 | [Paper](https://arxiv.org/pdf/2112.06905.pdf)                                                                                    |
|        PaLM        | 540B |     Decoder     |                                                                                                   -                                                                                                   | 2022-04 | [Paper](https://arxiv.org/pdf/2204.02311.pdf)                                                                                    |
|       MT-NLG       | 530B |     Decoder     |                                                                                                   -                                                                                                   | 2022-01 | [Paper](https://arxiv.org/pdf/2201.11990.pdf)                                                                                    |
|      J1-Jumbo      | 178B |     Decoder     |                                                                              [api](https://docs.ai21.com/docs/complete-api)                                                                              | 2021-08 | [Paper](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf)               |
|        OPT        | 175B |     Decoder     |                                                  [api](https://opt.alpa.ai) \| [ckpt](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)                                                  | 2022-05 | [Paper](https://arxiv.org/pdf/2205.01068.pdf)                                                                                    |
|       BLOOM       | 176B |     Decoder     |                                                      [api](https://huggingface.co/bigscience/bloom) \| [ckpt](https://huggingface.co/bigscience/bloom)                                                      | 2022-11 | [Paper](https://arxiv.org/pdf/2211.05100.pdf)                                                                                    |
|      GPT 3.0      | 175B |     Decoder     |                                                                                      [api](https://openai.com/api/)                                                                                      | 2020-05 | [Paper](https://arxiv.org/pdf/2005.14165.pdf)                                                                                    |
|       LaMDA       | 137B |     Decoder     |                                                                                                   -                                                                                                   | 2022-01 | [Paper](https://arxiv.org/pdf/2201.08239.pdf)                                                                                    |
|        GLM        | 130B |     Decoder     |                                                                                [ckpt](https://github.com/THUDM/GLM-130B)                                                                                | 2022-10 | [Paper](https://arxiv.org/pdf/2210.02414.pdf)                                                                                    |
|        YaLM        | 100B |     Decoder     |                                                                               [ckpt](https://github.com/yandex/YaLM-100B)                                                                               | 2022-06 | [Blog](https://medium.com/yandex/yandex-publishes-yalm-100b-its-the-largest-gpt-like-neural-network-in-open-source-d1df53d0e9a6) |
|       LLaMA       |  65B  |      Decoder      |                                                                          [ckpt](https://github.com/facebookresearch/llama)                                                                          | 2022-09 | [Paper](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/)                                                                                     |
|      GPT-NeoX      | 20B |     Decoder     |                                                                              [ckpt](https://github.com/EleutherAI/gpt-neox)                                                                              | 2022-04 | [Paper](https://arxiv.org/pdf/2204.06745.pdf)                                                                                    |
|        UL2        | 20B |    agnostic    | [ckpt](https://huggingface.co/google/ul2#:~:text=UL2%20is%20a%20unified%20framework%20for%20pretraining%20models,downstream%20fine-tuning%20is%20associated%20with%20specific%20pre-training%20schemes.) | 2022-05 | [Paper](https://arxiv.org/pdf/2205.05131v1.pdf)                                                                                  |
|    鹏程.盘古α    | 13B |     Decoder     |                                                      [ckpt](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/PanGu-α#模型下载)                                                      | 2021-04 | [Paper](https://arxiv.org/pdf/2104.12369.pdf)                                                                                    |
|         T5         | 11B | Encoder-Decoder |                                                                                  [ckpt](https://huggingface.co/t5-11b)                                                                                  | 2019-10 | [Paper](https://jmlr.org/papers/v21/20-074.html)                                                                                 |
|      CPM-Bee      | 10B |     Decoder     |                                                                                [api](https://live.openbmb.org/models/bee)                                                                                | 2022-10 | [Paper](https://arxiv.org/pdf/2012.00413.pdf)                                                                                    |
|       rwkv-4       |  7B  |      RWKV      |                                                                          [ckpt](https://huggingface.co/BlinkDL/rwkv-4-pile-7b)                                                                          | 2022-09 | [Github](https://github.com/BlinkDL/RWKV-LM)                                                                                     |
|       GPT-J       |  6B  |     Decoder     |                                                                            [ckpt](https://huggingface.co/EleutherAI/gpt-j-6B)                                                                            | 2022-09 | [Github](https://github.com/kingoflolz/mesh-transformer-jax)                                                                     |
|      GPT-Neo      | 2.7B |     Decoder     |                                                                              [ckpt](https://github.com/EleutherAI/gpt-neo)                                                                              | 2021-03 | [Github](https://github.com/EleutherAI/gpt-neo)                                                                                  |
|      GPT-Neo      | 1.3B |     Decoder     |                                                                              [ckpt](https://github.com/EleutherAI/gpt-neo)                                                                              | 2021-03 | [Github](https://github.com/EleutherAI/gpt-neo)                                                                                  |

### Instruction finetuned LLM
|       Model       | Size |  Architecture  |                                                                                               Access                                                                                               |  Date  | Origin                                                                                                                        |
| :----------------: | :--: | :-------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----: | ----------------------------------------------------------------------------------------------------------------------------- |
|Flan-PaLM| 540B | Decoder |-|2022-10|[Paper](https://arxiv.org/pdf/2210.11416.pdf)|
|BLOOMZ| 176B | Decoder | [ckpt](https://huggingface.co/bigscience/bloomz) |2022-11|[Paper](https://arxiv.org/pdf/2211.01786.pdf)|
| InstructGPT |175B| Decoder | [api](https://platform.openai.com/overview) | 2022-03 | [Paper](https://arxiv.org/pdf/2203.02155.pdf) |
|Galactica|120B|Decoder|[ckpt](https://huggingface.co/facebook/galactica-120b)|2022-11| [Paper](https://arxiv.org/pdf/2211.09085.pdf)|
| OpenChatKit| 20B | - |[ckpt](https://github.com/togethercomputer/OpenChatKit)| 2023-3 |-|
| Flan-UL2| 20B  | Decoder | [ckpt](https://github.com/google-research/google-research/tree/master/ul2)|2023-03 | [Blog](https://www.yitay.net/blog/flan-ul2-20b)|
| Gopher | - | - | - | - | - |
| Chinchilla | - | - | - | - |- |
|Flan-T5| 11B | Encoder-Decoder |[ckpt](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)|2022-10|[Paper](https://arxiv.org/pdf/2210.11416.pdf)|
|T0|11B|Encoder-Decoder|[ckpt](https://huggingface.co/bigscience/T0)|2021-10|[Paper](https://arxiv.org/pdf/2110.08207.pdf)|
|Alpaca| 7B|Decoder|[demo](https://crfm.stanford.edu/alpaca/)|2023-03|[Github](https://github.com/tatsu-lab/stanford_alpaca)|


### Aligned LLM
|       Model       | Size |  Architecture  |                                                                                               Access                                                                                               |  Date  | Origin                                                                                                                        |
| :----------------: | :--: | :-------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----: | ----------------------------------------------------------------------------------------------------------------------------- |
| GPT 4  | - | - | - | 2023-03 | [Blog](https://openai.com/research/gpt-4)|
|      ChatGPT      |  -  |     Decoder     |                                                                                 [demo](https://openai.com/blog/chatgpt/)\|[api](https://share.hsforms.com/1u4goaXwDRKC9-x9IvKno0A4sk30)   | 2022-11 | [Blog](https://openai.com/blog/chatgpt/)      |
| Sparrow  | 70B | - | - | 2022-09 | [Paper](https://arxiv.org/pdf/2209.14375.pdf)|
| Claude  | - | - | [demo](https://poe.com/claude)\|[api](https://www.anthropic.com/earlyaccess) | 2023-03 | [Blog](https://www.anthropic.com/index/introducing-claude) |

---

### Open LLM

- [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) - A foundational, 65-billion-parameter large language model. [LLaMA.cpp](https://github.com/ggerganov/llama.cpp) [Lit-LLaMA](https://github.com/Lightning-AI/lit-llama)
  - [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) - A model fine-tuned from the LLaMA 7B model on 52K instruction-following demonstrations. [Alpaca.cpp](https://github.com/antimatter15/alpaca.cpp) [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)
  - [Flan-Alpaca](https://github.com/declare-lab/flan-alpaca) - Instruction Tuning from Humans and Machines.
  - [Baize](https://github.com/project-baize/baize-chatbot) - Baize is an open-source chat model trained with [LoRA](https://github.com/microsoft/LoRA). It uses 100k dialogs generated by letting ChatGPT chat with itself. 
  - [Cabrita](https://github.com/22-hours/cabrita) - A portuguese finetuned instruction LLaMA.
  - [Vicuna](https://github.com/lm-sys/FastChat) - An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality. 
  - [Llama-X](https://github.com/AetherCortex/Llama-X) - Open Academic Research on Improving LLaMA to SOTA LLM.
  - [Chinese-Vicuna](https://github.com/Facico/Chinese-Vicuna) - A Chinese Instruction-following LLaMA-based Model.
  - [GPTQ-for-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa) - 4 bits quantization of [LLaMA](https://arxiv.org/abs/2302.13971) using [GPTQ](https://arxiv.org/abs/2210.17323).
  - [GPT4All](https://github.com/nomic-ai/gpt4all) - Demo, data, and code to train open-source assistant-style large language model based on GPT-J and LLaMa.
  - [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/) - A Dialogue Model for Academic Research
  - [BELLE](https://github.com/LianjiaTech/BELLE) - Be Everyone's Large Language model Engine
  - [StackLLaMA](https://huggingface.co/blog/stackllama) - A hands-on guide to train LLaMA with RLHF.
  - [RedPajama](https://github.com/togethercomputer/RedPajama-Data) -  An Open Source Recipe to Reproduce LLaMA training dataset.
  - [Chimera](https://github.com/FreedomIntelligence/LLMZoo) - Latin Phoenix.
- [BLOOM](https://huggingface.co/bigscience/bloom) - BigScience Large Open-science Open-access Multilingual Language Model [BLOOM-LoRA](https://github.com/linhduongtuan/BLOOM-LORA)
  - [BLOOMZ&mT0](https://huggingface.co/bigscience/bloomz) - a family of models capable of following human instructions in dozens of languages zero-shot.
  - [Phoenix](https://github.com/FreedomIntelligence/LLMZoo)
  
- [T5](https://arxiv.org/abs/1910.10683) - Text-to-Text Transfer Transformer 
  - [T0](https://arxiv.org/abs/2110.08207) - Multitask Prompted Training Enables Zero-Shot Task Generalization

- [OPT](https://arxiv.org/abs/2205.01068) - Open Pre-trained Transformer Language Models.
- [UL2](https://arxiv.org/abs/2205.05131v1) - a unified framework for pretraining models that are universally effective across datasets and setups. 
- [GLM](https://github.com/THUDM/GLM)- GLM is a General Language Model pretrained with an autoregressive blank-filling objective and can be finetuned on various natural language understanding and generation tasks.
  - [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) - ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数.
- [RWKV](https://github.com/BlinkDL/RWKV-LM) - Parallelizable RNN with Transformer-level LLM Performance.
  - [ChatRWKV](https://github.com/BlinkDL/ChatRWKV) - ChatRWKV is like ChatGPT but powered by my RWKV (100% RNN) language model.
- [StableLM](https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models) - Stability AI Language Models.
- [YaLM](https://medium.com/yandex/yandex-publishes-yalm-100b-its-the-largest-gpt-like-neural-network-in-open-source-d1df53d0e9a6) - a GPT-like neural network for generating and processing text. It can be used freely by developers and researchers from all over the world.
- [GPT-Neo](https://github.com/EleutherAI/gpt-neo) - An implementation of model & data parallel [GPT3](https://arxiv.org/abs/2005.14165)-like models using the [mesh-tensorflow](https://github.com/tensorflow/mesh) library.
- [GPT-J](https://github.com/kingoflolz/mesh-transformer-jax/#gpt-j-6b) - A 6 billion parameter, autoregressive text generation model trained on [The Pile](https://pile.eleuther.ai/).
  - [Dolly](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html) - a cheap-to-build LLM that exhibits a surprising degree of the instruction following capabilities exhibited by ChatGPT.

- [Pythia](https://github.com/EleutherAI/pythia) - Interpreting Autoregressive Transformers Across Time and Scale
  - [Dolly 2.0](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) - the first open source, instruction-following LLM, fine-tuned on a human-generated instruction dataset licensed for research and commercial use.
- [OpenFlamingo](https://github.com/mlfoundations/open_flamingo) - an open-source reproduction of DeepMind's Flamingo model.
- [Cerebras-GPT](https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/) - A Family of Open, Compute-efficient, Large Language Models.
- [GALACTICA](https://github.com/paperswithcode/galai/blob/main/docs/model_card.md) - The GALACTICA models are trained on a large-scale scientific corpus.
  - [GALPACA](https://huggingface.co/GeorgiaTechResearchInstitute/galpaca-30b) - GALACTICA 30B fine-tuned on the Alpaca dataset.

- [Palmyra](https://huggingface.co/Writer/palmyra-base) - Palmyra Base was primarily pre-trained with English text.
- [Camel](https://huggingface.co/Writer/camel-5b-hf) - a state-of-the-art instruction-following large language model designed to deliver exceptional performance and versatility.
- [h2oGPT](https://github.com/h2oai/h2ogpt)
- [PanGu-α](https://openi.org.cn/pangu/) - PanGu-α is a 200B parameter autoregressive pretrained Chinese language model develped by Huawei Noah's Ark Lab, MindSpore Team and Peng Cheng Laboratory.
- [MOSS](https://github.com/OpenLMLab/MOSS) - MOSS是一个支持中英双语和多种插件的开源对话语言模型.
- [Open-Assistant](https://github.com/LAION-AI/Open-Assistant) - a project meant to give everyone access to a great chat based large language model.
  - [HuggingChat](https://huggingface.co/chat/) - Powered by Open Assistant's latest model – the best open source chat model right now and @huggingface Inference API.

## Others

- [Evaluating Language Models by OpenAI, DeepMind, Google, Microsoft](https://levelup.gitconnected.com/how-to-benchmark-language-models-by-openai-deepmind-google-microsoft-783d4307ec50) - Evaluating Language Models by OpenAI, DeepMind, Google, Microsoft.

## Other Awesome Lists

- [Awesome LLM](https://github.com/Hannibal046/Awesome-LLM/) -  A curated list of papers about large language models. 
- [Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts) - A collection of prompt examples to be used with the ChatGPT model.
- [awesome-chatgpt-prompts-zh](https://github.com/PlexPt/awesome-chatgpt-prompts-zh) - A Chinese collection of prompt examples to be used with the ChatGPT model.
- [Awesome ChatGPT](https://github.com/humanloop/awesome-chatgpt) - Curated list of resources for ChatGPT and GPT-3 from OpenAI.
- [Chain-of-Thoughts Papers](https://github.com/Timothyxxx/Chain-of-ThoughtsPapers) -  A trend starts from "Chain of Thought Prompting Elicits Reasoning in Large Language Models.
- [Instruction-Tuning-Papers](https://github.com/SinclairCoder/Instruction-Tuning-Papers) - A trend starts from `Natrural-Instruction` (ACL 2022), `FLAN` (ICLR 2022) and `T0` (ICLR 2022).
- [LLM Reading List](https://github.com/crazyofapple/Reading_groups/) - A paper & resource list of large language models.
- [Reasoning using Language Models](https://github.com/atfortes/LM-Reasoning-Papers) - Collection of papers and resources on Reasoning using Language Models.
- [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub) - Measuring LLMs' Reasoning Performance
- [Awesome GPT](https://github.com/formulahendry/awesome-gpt) - A curated list of awesome projects and resources related to GPT, ChatGPT, OpenAI, LLM, and more.
- [Awesome GPT-3](https://github.com/elyase/awesome-gpt3) - a collection of demos and articles about the [OpenAI GPT-3 API](https://openai.com/blog/openai-api/).
