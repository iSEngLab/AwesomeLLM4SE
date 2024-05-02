# Large Language Models for Software Engineering
>*Title*: [A Survey on Large Language Models for Software Engineering](https://arxiv.org/abs/2312.15223)
>
>*Authors*: [Quanjun Zhang](https://sites.google.com/view/quanjunzhang/), [Chunrong Fang](https://chunrong.github.io/), Yang Xie, Yaxin Zhang, [Yun Yang](https://www.swinburne.edu.au/research/our-research/access-our-research/find-a-researcher-or-supervisor/researcher-profile/?id=yyang), [Weisong Sun](https://sites.google.com/view/wssun/), [Shengcheng Yu](https://www.seysc.com/), [Zhenyu Chen](https://scholar.google.com.au/citations?user=HQWxCnkAAAAJ&hl=zh-CN&oi=sra)


A collection of academic publications and methodologies on the classification of Code Large Language Models' pre-training tasks, downstream tasks, and the application of **Large Language Models** in the field of **Software Engineering.**

We welcome all researchers to contribute to this repository and further contribute to the knowledge of the Large Language Models with Software Engineering field.
Please feel free to contact us if you have any related references by Github issue or pull request. 

------

## Citation

Please read and cite our paper:

```
@article{zhang2023survey,
  title={A Survey on Large Language Models for Software Engineering},
  author={Zhang, Quanjun and Fang, Chunrong and Xie, Yang and Zhang, Yaxin and Yang, Yun and Sun, Weisong and Yu, Shengcheng and Chen, Zhenyu},
  journal={arXiv preprint arXiv:2312.15223},
  year={2023}
}
```

------

## Code LLMs

| Model         | Publisher | Architecture    | Data Resource                                               | Public | URL                                                          | Title                                                        |
| ------------- | --------- | --------------- | ----------------------------------------------------------- | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| CommitBERT    | ACL       | Encoder-only    | CodeSearchNet                                               | √      | [GitHub](https://github.com/graykode/commit-autosuggestions) | [Commitbert: Commit message generation using pre-trained programming language model](https://arxiv.org/abs/2105.14242) |
| CuBERT        | ICML      | Encoder-only    | Github Code                                                 | √      | [GitHub](https://github.com/google-research/google-research/tree/master/cubert) | [Learning and evaluating contextual embedding of source code](https://proceedings.mlr.press/v119/kanade20a.html) |
| CodeBERT      | EMNLP     | Encoder-only    | CodeSearchNet, Codenn                                       | √      | [GitHub](https://github.com/microsoft/CodeBERT)              | [Codebert: A pre-trained model for programming and natural languages](https://arxiv.org/abs/2002.08155) |
| GraphCodeBERT | ICLR      | Encoder-only    | CodeSearchNet, Tufano, BigCloneBench                        | √      | [GitHub](https://github.com/microsoft/CodeBERT)              | [Graphcodebert: Pre-training code representations with data flow](https://arxiv.org/abs/2009.08366) |
| UnixCoder     | ACL       | Encoder-Decoder |                                                             | √      |                                                              | [Unixcoder: Unified cross-modal pre-training for code representation](https://arxiv.org/abs/2203.03850) |
| JuPyT5        | arXiv     | Encoder-Decoder | Human Eval                                                  | √      | [GitHub](https://github.com/microsoft/DataScienceProblems)   | [Training and evaluating a jupyter notebook data science assistant](https://arxiv.org/abs/2201.12901) |
| CodeT5Mix     | arXiv     | Encoder-Decoder | CodeSearchNet, CodeParrot                                   | √      | -                                                            | [CodeT5Mix: A Pretrained Mixture of Encoder-decoder Transformers for Code Understanding and Generation](https://openreview.net/forum?id=VPCi3STZcaO) |
| ERNIE-Code    | arXiv     | Encoder-Decoder | CodeSearchNet, CC-100, OPUS, MultiUN, IIT, WikiMatrix       | √      | [GitHub](https://github.com/PaddlePaddle/PaddleNLP)          | [ERNIE-Code Beyond English-Centric Cross-lingual Pretraining for Programming Languages](https://arxiv.org/abs/2212.06742) |
| CodeT5+       | arXiv     | Encoder-Decoder | CodeSearchNet, Github Code, Human Eval, MathQAPython, GSM8K | √      | [GitHub](https://github.com/salesforce/CodeT5/tree/main/CodeT5%2B) | [CodeT5+: Open Code Large Language Models for Code Understanding and Generation](https://arxiv.org/abs/2305.07922) |
| CoditT5       | ASE       | Encoder-Decoder | CodeSearchNet                                               |        | [GitHub](https://github.com/engineeringsoftware/coditt5)     | [CoditT5: Pretraining for Source Code and Natural Language Editing](https://dl.acm.org/doi/abs/10.1145/3551349.3556955) |
| PyMT5         | EMNLP     | Encoder-Decoder | CodeSearchNet                                               | ×      | [GitHub](https://github.com/devcartel/pymt5)                 | [PyMT5 multi-mode translation of natural language and Python code with transformers](https://arxiv.org/abs/2010.03150) |
| CodeT5        | EMNLP     | Encoder-Decoder | CodeSearchNet, BigQuery                                     | √      | [GitHub](https://github.com/salesforce/CodeT5)               | [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation](https://arxiv.org/abs/2109.00859) |
| SPT-Code      | ICSE      | Encoder-Decoder | CodeSearchNet, JCSD, PCSD, Tufano, Alon                     | √      | [GitHub](https://github.com/NougatCA/SPT-Code)               | [Spt-code: Sequence-to-sequence pre-training for learning source code representations](https://dl.acm.org/doi/abs/10.1145/3510003.3510096) |
| PLBART        | NAACL     | Encoder-Decoder | CodeSearchNet, BigQuery, Tufano, CodeXGLUE, Concode, Devign | √      | [GitHub](https://github.com/wasiahmad/PLBART)                | [Unified Pre-training for Program Understanding and Generation](https://arxiv.org/abs/2103.06333) |
| CodeRL        | NeurIPS   | Encoder-Decoder | CodeSearchNet, APPS, MBPP                                   | √      | [GitHub](https://github.com/salesforce/CodeRL)               | [CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning]([CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning (neurips.cc)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8636419dea1aa9fbd25fc4248e702da4-Abstract-Conference.html)) |
| AlphaCode     | Science   | Encoder-Decoder | CodeContests                                                | ×      |                                                              | [Competition-Level Code Generation with AlphaCode](https://www.science.org/doi/abs/10.1126/science.abq1158) |
| T5-Learning   | ICSE      | Encoder-Decoder | CodeSearch                                                  | √      | [GitHub](https://github.com/wasiahmad/PLBART)                | [Unified pre-training for program understanding and generation](https://arxiv.org/abs/2103.06333) |
| Codex         | arXiv     | Decoder-only    | Human Eval                                                  | ×      | [GitHub](https://github.com/adrianhajdin/project_openai_codex) | [Evaluating large language models trained on code](https://arxiv.org/abs/2107.03374) |
| PaLM-Coder    | arXiv     | Decoder-only    | Github Code                                                 | ×      | [GitHub](https://github.com/lucidrains/PaLM-rlhf-pytorch)    | [PaLM Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) |
| PanGu-Coder   | arXiv     | Decoder-only    | Human Eval, MBPP                                            | √      | [GitHub](https://github.com/baurine/vscode-pangu)            | [PanGu-Coder Program Synthesis with Function-Level Language Modeling](https://arxiv.org/abs/2207.11280) |
| BLOOM         | arXiv     | Decoder-only    | Human Eval, ThePile                                         | √      | [GitHub](https://github.com/bigscience-workshop/bigscience)  | [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100) |
| CodeGeeX      | arXiv     | Decoder-only    | ThePile, CodeParrot                                         | √      | [GitHub](https://github.com/THUDM/CodeGeeX)                  | [CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X](https://arxiv.org/abs/2303.17568) |
| SantaCoder    | arXiv     | Decoder-only    | The Stack v1.1                                              | √      |                                                              | [SantaCoder don’t reach for the stars!](https://arxiv.org/abs/2301.03988) |
| StarCoder     | arXiv     | Decoder-only    | The Stack v1.1                                              | √      |                                                              | [StarCoder may the source be with you](https://arxiv.org/abs/2305.06161) |
| CodeGen2      | arXiv     | Decoder-only    | Human Eval                                                  | √      |                                                              | [CodeGen2 Lessons for Training LLMs on Programming and Natural Languages](https://arxiv.org/abs/2305.02309) |
| GPT-C         | FSE       | Decoder-only    |                                                             | √      |                                                              | [IntelliCode compose code generation using transformer](https://dl.acm.org/doi/abs/10.1145/3368089.3417058) |
| PolyCoder     | ICLR      | Decoder-only    | CodeSearchNet, BigQuery, Human Eval, ThePile, CodeParrot    | √      | [GitHub](https://github.com/VHellendoorn/Code-LMs)           | [A Systematic Evaluation of Large Language Models of Code](https://dl.acm.org/doi/abs/10.1145/3520312.3534862) |
| CodeGen       | ICLR      | Decoder-only    | BigQuery, Human Eval, ThePile, BigPython                    | √      | [GitHub](https://github.com/salesforce/CodeGen)              | [CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis](https://arxiv.org/abs/2203.13474) |
| InCoder       | ICLR      | Decoder-only    | Human Eval, CodeXGLUE, TypeWriter OSS                       | √      | [Website](https://sites.google.com/view/incoder-code-models) | [InCoder: A Generative Model for Code Infilling and Synthesis](https://arxiv.org/abs/2204.05999) |
| PyCodeGPT     | IJCAI     | Decoder-only    | Github Code, Human Eval, CodeXGLUE                          | √      | [GitHub](https://github.com/microsoft/pycodegpt)             | [CERT Continual Pre-Training on Sketches for Library-Oriented Code Generation](https://arxiv.org/abs/2206.06888) |
| CodeGPT       | NeurIPS   | Decoder-only    | CodeSearchNet                                               | √      | [GitHub](https://github.com/appleboy/CodeGPT)                | [CodeXGLUE A Machine Learning Benchmark Dataset for Code Understanding and Generation](https://arxiv.org/abs/2102.04664) |
| GPT-Neo       | -         | Decoder-only    | ThePile                                                     | √      | [GitHub](https://github.com/EleutherAI/gpt-neo)              |                                                              |
| GPT-CC        | -         | Decoder-only    | Human Eval, APPS, ThePile, Code Clippy Data                 | √      | [GitHub](https://github.com/CodedotAl/gpt-code-clippy/wiki)  |                                                              |
| GPT-J         | -         | Decoder-only    | ThePile                                                     | √      | [GitHub](https://github.com/kingoflolz/mesh-transformer-jax) |                                                              |
| PanGu-Coder2  | arXiv     | Decoder-only    | Human Eval, MBPP                                            | ×      |                                                              | [Pangu-coder2: Boosting large language models for code with ranking feedback](https://arxiv.org/abs/2307.14936) |

------

## LLM4SE

| Task                               | Paper Title                                                  | Year | Publisher   |
| ---------------------------------- | ------------------------------------------------------------ | ---- | ----------- |
| Software Specifications Generation | [Impact of Large Language Models on Generating Software Specifications](https://arxiv.org/abs/2306.03324) | 2023 | arXiv       |
| Requirements classification        | [NoRBERT: Transfer Learning for Requirements Classification](https://ieeexplore.ieee.org/abstract/document/9218141) | 2020 | RE          |
| Requirements classification        | [Improving Requirements Classification Models Based on Explainable Requirements Concerns ](https://ieeexplore.ieee.org/abstract/document/10260874) | 2023 | REW         |
| Requirements classification        | [Pre-Trained Model-Based NFR Classification: Overcoming Limited Data Challenges](https://ieeexplore.ieee.org/abstract/document/10207690) | 2023 | IEEE Access |
| Requirements classification        | [Non Functional Requirements Identification and Classification Using Transfer Learning Model](https://ieeexplore.ieee.org/abstract/document/10181313) | 2023 | IEEE Access |
| Requirements classification        | [BERT-Based Approach for Greening Software Requirements Engineering Through Non-Functional Requirements](https://ieeexplore.ieee.org/abstract/document/10256174) | 2023 | IEEE Access |
| GUI Layouts                        | [Evaluating a Large Language Model on Searching for GUI Layouts](https://dl.acm.org/doi/abs/10.1145/3593230) |      | EICS        |
| Code Generation                    | [AceCoder: Utilizing Existing Code to Enhance Code Generation](https://arxiv.org/abs/2303.17780) | 2023 | arXiv       |
| Code Generation                    | [A Study on Robustness and Reliability of Large Language Model Code Generation](https://arxiv.org/abs/2308.10335v1) | 2023 | arXiv       |
| Code Generation                    | [Self-Edit: Fault-Aware Code Editor for Code Generation](https://arxiv.org/abs/2305.04087) | 2023 | arXiv       |
| Code Generation                    | [ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation](https://arxiv.org/abs/2308.01861) | 2023 | arXiv       |
| Code Generation                    | [A Syntax-Guided Multi-Task Learning Approach for Turducken-Style Code Generation](https://arxiv.org/abs/2303.05061) | 2023 | arXiv       |
| Code Generation                    | [Improving ChatGPT Prompt for Code Generation](https://arxiv.org/abs/2305.08360) | 2023 | arXiv       |
| Code Generation                    | [LEVER: Learning to Verify Language-to-Code Generation with Execution](https://arxiv.org/abs/2302.08468) | 2023 | ICML        |
| Code Generation                    | [Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation](https://arxiv.org/abs/2305.01210) | 2023 | arXiv       |
| Code Generation                    | [Is Model Attention Aligned with Human Attention? An Empirical Study on Large Language Models for Code Generation](https://arxiv.org/abs/2306.01220#:~:text=An analysis of five LLMs on a popular,of LLMs and their alignment with human programmers.) | 2023 | arXiv       |
| Code Generation                    | [ClarifyGPT: Empowering LLM-based Code Generation with Intention Clarification](https://arxiv.org/abs/2310.10996) | 2023 | arXiv       |
| Code Generation                    | [Self-collaboration Code Generation via ChatGPT](https://arxiv.org/abs/2304.07590) | 2023 | arXiv       |
| Code Generation                    | [LLM is Like a Box of Chocolates: the Non-determinism of ChatGPT in Code Generation](https://arxiv.org/abs/2308.02828) | 2023 | arXiv       |
| Code Search                        | [CodeTF: One-stop Transformer Library for State-of-the-art Code LLM](https://arxiv.org/abs/2306.00029) | 2023 | arXiv       |
| Code Search                        | [CodeRetriever: A Large Scale Contrastive Pre-Training Method for Code Search](https://aclanthology.org/2022.emnlp-main.187/) | 2022 | EMNLP       |
| Code Search                        | [On Contrastive Learning of Semantic Similarity forCode to Code Search](https://arxiv.org/abs/2305.03843) | 2023 | arXiv       |
| Code Search                        | [CCT-Code: Cross-Consistency Training for Multilingual Clone Detection and Code Search](https://arxiv.org/abs/2305.11626) | 2023 | arXiv       |
| Code Search                        | [On the Effectiveness of Transfer Learning for Code Search](https://ieeexplore.ieee.org/abstract/document/9835142) | 2022 | TSE         |
| Code Translation                   | [Multilingual Code Co-Evolution Using Large Language Models](https://arxiv.org/abs/2307.14991) | 2023 | arXiv       |
| Code Translation                   | [Learning Transfers over Several Programming Languages](https://arxiv.org/abs/2310.16937) | 2023 | arXiv       |
| Bug Generation                     | [Automated Bug Generation in the era of Large Language Models](https://arxiv.org/abs/2310.02407) | 2023 | arXiv       |
| Code Comment Completion            | [Automated Bug Generation in the era of Large Language Models](https://arxiv.org/abs/2310.02407) | 2021 | ICSME       |
| Code Summarization                 | [Achieving High-Level Software Component Summarization via Hierarchical Chain-of-Thought Prompting and Static Code Analysis](https://ieeexplore.ieee.org/abstract/document/10292037) | 2023 | ICoDSE      |
| Code Summarization                 | [Automatic Code Summarization via ChatGPT: How Far Are We?](https://arxiv.org/abs/2305.12865) | 2023 | arXiv       |
| Code Completion                    | [An Empirical Study on the Usage of Transformer Models for Code Completion](https://ieeexplore.ieee.org/abstract/document/9616462) | 2021 | TSE         |
| Code Completion                    | [Enriching Source Code with Contextual Data for Code Completion Models: An Empirical Study](https://arxiv.org/abs/2304.12269) | 2023 | arXiv       |
| Code Completion                    | [CCTEST: Testing and Repairing Code Completion Systems](https://ieeexplore.ieee.org/abstract/document/10172845) | 2023 | ICSE        |
| Code Completion                    | [An Empirical Study on the Usage of BERT Models for Code Completion](https://ieeexplore.ieee.org/abstract/document/9463129) | 2021 | MSR         |
| Code Completion                    | [Learning Deep Semantics for Test Completion](https://arxiv.org/abs/2302.10166) | 2023 | arXiv       |
| Program Synthesis                  | [Jigsaw: Large language models meet program synthesis](https://dl.acm.org/doi/abs/10.1145/3510003.3510203) | 2022 | ICSE        |
| Program Synthesis                  | [Fully Autonomous Programming with Large Language Models](https://arxiv.org/abs/2304.10423) | 2023 | arXiv       |
| Code Edits Prediction              | [GrACE: Generation using Associated Code Edits](https://arxiv.org/abs/2305.14129) | 2023 | arXiv       |
| Fault Localization                 | [Fast changeset-based bug localization with BERT](https://dl.acm.org/doi/abs/10.1145/3510003.3510042) | 2022 | ICSE        |
| Fault Localization                 | [Enhancing Bug Localization Using Phase-Based Approach](https://ieeexplore.ieee.org/abstract/document/10097736) | 2023 | arXiv       |
| Fault Localization                 | [Large Language Models in Fault Localisation (arxiv.org)](https://arxiv.org/abs/2308.15276) | 2023 | arXiv       |
| Fault Localization                 | [A Preliminary Evaluation of LLM-Based Fault Localization](https://arxiv.org/abs/2308.05487) | 2023 | arXiv       |
| Fault Localization                 | [TroBo: A Novel Deep Transfer Model for Enhancing Cross-Project Bug Localization](https://link.springer.com/chapter/10.1007/978-3-030-82136-4_43) | 2021 | KSEM        |
| Decompilation                      | [LmPa: Improving Decompilation by Synergy of Large Language Model and Program Analysis](https://arxiv.org/abs/2306.02546) | 2023 | arXiv       |
| Decompilation                      | [Refining Decompiled C Code with Large Language Models](https://arxiv.org/abs/2310.06530) | 2023 | arXiv       |
| Vulnerability Prediction           | [LineVul- A Transformer-based Line-Level Vulnerability Prediction](https://dl.acm.org/doi/abs/10.1145/3524842.3528452) | 2021 | MSR         |
| Vulnerability Prediction           | [An Empirical Study of Deep Learning Models for Vulnerability Detection](https://ieeexplore.ieee.org/abstract/document/10172583) | 2023 | ICSE        |
| Vulnerability Detection            | [Low Level Source Code Vulnerability Detection Using Advanced BERT Language Model](https://caiac.pubpub.org/pub/gdhb8oq4/release/1) | 2022 | Canadian AI |
| Vulnerability Detection            | [FLAG: Finding Line Anomalies (in code) with Generative AI](https://arxiv.org/abs/2306.12643) | 2023 | arXiv       |
| Vulnerability Detection            | [Prompt-Enhanced Software Vulnerability Detection Using ChatGPT](https://arxiv.org/abs/2308.12697) | 2023 | arXiv       |
| Vulnerability Detection            | [ZeroLeak: Using LLMs for Scalable and Cost Effective Side-Channel Patching](https://arxiv.org/abs/2308.13062) | 2023 | arXiv       |
| Test Generation                    | [Generating Accurate Assert Statements for Unit Test Cases using Pretrained Transformers](https://dl.acm.org/doi/abs/10.1145/3524481.3527220) | 2022 | AST         |
| Test Generation                    | [An Empirical Evaluation of Using Large Language Models for Automated Unit Test Generation](https://arxiv.org/abs/2302.06527) | 2023 | arXiv       |
| Test Generation                    | [ChatUniTest: a ChatGPT-based automated unit test generation tool](https://arxiv.org/abs/2305.04764) | 2023 | arXiv       |
| Test Generation                    | [ChatGPT vs SBST: A comparative assessment of unit test suite generation](https://arxiv.org/abs/2307.00588) | 2023 | arXiv       |
| Test Generation                    | [Unit Test Case Generation with Transformers and Focal Context](https://arxiv.org/abs/2009.05617) | 2020 | arXiv       |
| Test Generation                    | [CODAMOSA: Escaping Coverage Plateaus in Test Generation with Pre-trained Large Language Models](https://ieeexplore.ieee.org/document/10172800) | 2023 | ICSE        |
| Assertion Generation               | [TOGA: A Neural Method for Test Oracle Generation](https://dl.acm.org/doi/abs/10.1145/3510003.3510141) | 2022 | ICSE        |
| Test Minimization                  | [LTM: Scalable and Black-box Similarity-based Test Suite Minimization based on Language Models](https://arxiv.org/abs/2304.01397) | 2023 | arXiv       |
| Fuzzing                            | [SearchGEM5: Towards Reliable gem5 with Search Based Software Testing and Large Language Models](https://kclpure.kcl.ac.uk/portal/en/publications/searchgem5-towards-reliable-gem5-with-search-based-software-testi) | 2023 | SSBSE       |
| Fuzzing                            | [Augmenting Greybox Fuzzing with Generative AI](https://arxiv.org/abs/2306.06782) | 2023 | arXiv       |
| Fuzzing                            | [White-box Compiler Fuzzing Empowered by Large Language Models](https://arxiv.org/abs/2310.15991) | 2023 | arXiv       |
| Fuzzing                            | [Large Language Model guided Protocol Fuzzing]([NDSS24.pdf (mboehme.github.io)](https://mboehme.github.io/paper/NDSS24.pdf)) | 2023 | -           |
| Fuzzing                            | [Large Language Models are Zero-Shot Fuzzers: Fuzzing Deep-Learning Libraries via Large Language Models](https://dl.acm.org/doi/abs/10.1145/3597926.3598067) | 2023 | ISSTA       |
| Fuzzing                            | [Universal Fuzzing via Large Language Models](https://arxiv.org/abs/2308.04748) | 2023 | arXiv       |
| Fuzzing                            | [Large Language Models are Edge-Case Fuzzers: Testing Deep Learning Libraries via FuzzGPT](https://arxiv.org/abs/2304.02014) | 2023 | arXiv       |
| Property-based Testing             | [Can Large Language Models Write Good Property-Based Tests?](https://arxiv.org/abs/2307.04346) | 2023 | arXiv       |
| Failure-Inducing Testing           | [Nuances are the Key: Unlocking ChatGPT to Find Failure-Inducing Tests with Differential Prompting](https://arxiv.org/abs/2304.11686) | 2023 | arXiv       |
| Penetration Testing                | [Getting pwn'd by AI: Penetration Testing with Large Language Models](https://arxiv.org/abs/2308.00121) | 2023 | arXiv       |
| Penetration Testing                | [PentestGPT: An LLM-empowered Automatic Penetration Testing Tool](https://arxiv.org/abs/2308.06782) | 2023 | arXiv       |
| Mutation Testing                   | [Efficient Mutation Testing via Pre-Trained Language Models](https://arxiv.org/abs/2301.03543) | 2023 | arXiv       |
| Mutation Testing                   | [Effective Test Generation Using Pre-trained Large Language Models and Mutation Testing](https://arxiv.org/abs/2308.16557) | 2023 | arXiv       |
| Mutation Testing                   | [Automated Bug Generation in the era of Large Language Models](https://arxiv.org/abs/2310.02407) | 2023 | arXiv       |
| Mutation Testing                   | [VULGEN: Realistic Vulnerability Generation Via Pattern Mining and Deep Learning](https://www.software-lab.org/publications/icse2023_VulGen.pdf) | 2023 | ICSE        |
| GUI Testing                        | [Fill in the Blank: Context-aware Automated Text Input Generation for Mobile GUI Testing](https://ieeexplore.ieee.org/abstract/document/10172490) | 2022 | ICSE        |
| GUI Testing                        | [Make LLM a Testing Expert: Bringing Human-like Interaction to Mobile GUI Testing via Functionality-aware Decisions](https://arxiv.org/abs/2310.15780) | 2023 | arXiv       |
| NLP Testing                        | [Improving Machine Translation Systems via Isotopic Replacement](https://dl.acm.org/doi/abs/10.1145/3510003.3510206) | 2022 | ICSE        |
| NLP Testing                        | [Structure-Invariant Testing for Machine Translation](https://ieeexplore.ieee.org/document/9284002) | 2020 | ICSE        |
| NLP Testing                        | [Automated Testing for Machine Translation via Constituency Invariance](https://ieeexplore.ieee.org/abstract/document/9678715) | 2021 | ASE         |
| NLP Testing                        | [Machine Translation Testing via Pathological Invariance](https://dl.acm.org/doi/abs/10.1145/3368089.3409756) | 2020 | ESEC/FSE    |
| Code Review                        | [Automating code review activities by large-scale pre-training](https://dl.acm.org/doi/abs/10.1145/3540250.3549081) | 2022 | FSE         |
| Code Review                        | [AUGER: automatically generating review comments with pre-training models](https://dl.acm.org/doi/abs/10.1145/3540250.3549099) | 2022 | FSE         |
| Code Review                        | [Using Pre-Trained Models to Boost Code Review Automation](https://dl.acm.org/doi/abs/10.1145/3510003.3510621) | 2022 | ICSE        |
| Code Review                        | [Exploring the Potential of ChatGPT in Automated Code Refinement: An Empirical Study](https://arxiv.org/abs/2309.08221) | 2023 | arXiv       |
| Duplicate Bug Report Detection     | [Cupid: Leveraging ChatGPT for More Accurate Duplicate Bug Report Detection](https://arxiv.org/abs/2308.10022) | 2023 | arXiv       |
| Duplicate Bug Report Detection     | [Can LLMs Demystify Bug Reports?](https://arxiv.org/abs/2310.06310) | 2023 | arXiv       |
| Bug Reproduction                   | [Large Language Models are Few-shot Testers: Exploring LLM-based General Bug Reproduction](https://ieeexplore.ieee.org/abstract/document/10172763) | 2023 | ICSE        |
| Program Repair                     | [CURE Code-Aware Neural Machine Translation for Automatic Program Repair](https://ieeexplore.ieee.org/abstract/document/9401997) | 2021 | ICSE        |
| Program Repair                     | [DEAR A Novel Deep Learning-based Approach for Automated Program Repair](https://dl.acm.org/doi/abs/10.1145/3510003.3510177) | 2022 | ICSE        |
| Program Repair                     | [CIRCLE: Continual repair across programming languages](https://dl.acm.org/doi/abs/10.1145/3533767.3534219) | 2022 | ISSTA       |
| Program Repair                     | [InferFix: End-to-End Program Repair with LLMs](https://arxiv.org/abs/2303.07263) | 2023 | arXiv       |
| Program Repair                     | [TFix: Learning to Fix Coding Errors with a Text-to-Text Transformer](http://proceedings.mlr.press/v139/berabi21a.html) | 2021 | ICML        |
| Program Repair                     | [Less Training, More Repairing Please: Revisiting Automated Program Repair via Zero-shot Learning](https://dl.acm.org/doi/abs/10.1145/3540250.3549101) | 2022 | FSE         |
| Program Repair                     | [CopilotingtheCopilots: Fusing Large Language Models with Completion Engines for Automated Program Repair](https://arxiv.org/abs/2309.00608) | 2023 | arXiv       |
| Program Repair                     | [Revisiting the Plastic Surgery Hypothesis via Large Language Models](https://arxiv.org/abs/2303.10494) | 2023 | arXiv       |
| Program Repair                     | [Automated Program Repair in the Era of Large Pre-trained Language Models](https://ieeexplore.ieee.org/document/10172803) | 2023 | ICSE        |
| Program Repair                     | [Impact of Code Language Models on Automated Program Repair](https://dl.acm.org/doi/10.1109/ICSE48619.2023.00125) | 2023 | ICSE        |
| Program Repair                     | [Generating Bug-Fixes Using Pretrained Transformers](https://dl.acm.org/doi/abs/10.1145/3460945.3464951) | 2021 | PLDI        |
| Program Repair                     | [Keep the Conversation Going- Fixing 162 out of 337 bugs for $0.42 each using ChatGPT](https://arxiv.org/abs/2304.00385) | 2023 | arXiv       |
| Program Repair                     | [An Analysis of the Automatic Bug Fixing Performance of ChatGPT](https://arxiv.org/abs/2301.08653) | 2023 | arXiv       |
| Program Repair                     | [Automated Repair of Programs from Large Language Models](https://ieeexplore.ieee.org/abstract/document/10172854) | 2022 | ICSE        |
| Program Repair                     | [A study on Prompt Design, Advantages and Limitations of ChatGPT for Deep Learning Program Repair](https://arxiv.org/abs/2304.08191) | 2023 | arXiv       |
| Program Repair                     | [Gamma: Revisiting template-based automated program repair via mask prediction](https://arxiv.org/abs/2309.09308) | 2023 | arXiv       |
| Program Repair                     | [A Critical Review of Large Language Model on Software Engineering: An Example from ChatGPT and Automated Program Repair](https://arxiv.org/abs/2310.08879) | 2023 | arXiv       |
| Program Repair                     | [Domain Knowledge Matters: Improving Prompts with Fix Templates for Repairing Python Type Errors](https://arxiv.org/abs/2306.01394) | 2023 | arXiv       |
| Vulnerability Repair               | [VulRepair: A T5-Based Automated Software Vulnerability Repair](https://dl.acm.org/doi/abs/10.1145/3540250.3549098) | 2022 | FSE         |
| Vulnerability Repair               | [Examining Zero-Shot Vulnerability Repair with Large Language Models](https://ieeexplore.ieee.org/abstract/document/10179324) | 2023 | SP          |
| Vulnerability Repair               | [Can Large Language Models Find And Fix Vulnerable Software?](https://arxiv.org/abs/2308.10345) | 2023 | arXiv       |
| Vulnerability Repair               | [Pre-Trained Model-Based Automated Software Vulnerability Repair: How Far are We?](https://ieeexplore.ieee.org/abstract/document/10232867) | 2023 | TDSC        |
| Patch Correctness Assessment       | [Evaluating representation learning of code changes for predicting patch correctness in program repair](https://dl.acm.org/doi/abs/10.1145/3324884.3416532) | 2020 | ASE         |
| Patch Correctness Assessment       | [The Best of Both Worlds: Combining Learned Embeddings with Engineered Features for Accurate Prediction of Correct Patches](https://dl.acm.org/doi/full/10.1145/3576039) | 2023 | TOSEM       |
| Patch Correctness Assessment       | [Is this Change the Answer to that Problem? Correlating Descriptions of Bug and Code Changes for Evaluating Patch Correctness](https://dl.acm.org/doi/abs/10.1145/3551349.3556914) | 2022 | ASE         |
| Bug Replay                         | [Prompting Is All Your Need: Automated Android Bug Replay with Large Language Models](https://arxiv.org/abs/2306.01987) | 2023 | arXiv       |







