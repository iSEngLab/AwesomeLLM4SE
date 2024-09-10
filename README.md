<h1 align = "center">Large Language Models for Software Engineering</h1>
<p align="center">
  <a href="https://arxiv.org/abs/2312.15223"><img src="https://img.shields.io/badge/arXiv-2405.01466-blue.svg"></a>
  <img src="https://img.shields.io/github/stars/iSEngLab/AwesomeLLM4SE?color=yellow&label=Stars">
</p>
<p align="center">
  <a href="https://github.com/iSEngLab/AwesomeLLM4SE">View on GitHub</a>
</p>

>*Title*: [A Survey on Large Language Models for Software Engineering](https://arxiv.org/abs/2312.15223)
>
>*Authors*: [Quanjun Zhang](https://sites.google.com/view/quanjunzhang/), [Chunrong Fang](https://chunrong.github.io/), Yang Xie, Yaxin Zhang, [Yun Yang](https://www.swinburne.edu.au/research/our-research/access-our-research/find-a-researcher-or-supervisor/researcher-profile/?id=yyang), [Weisong Sun](https://sites.google.com/view/wssun/), [Shengcheng Yu](https://www.seysc.com/), [Zhenyu Chen](https://scholar.google.com.au/citations?user=HQWxCnkAAAAJ&hl=zh-CN&oi=sra)

A collection of academic publications and methodologies on the classification of Code Large Language Models' pre-training tasks, downstream tasks, and the application of **Large Language Models** in the field of **Software Engineering.**

We welcome all researchers to contribute to this repository and further contribute to the knowledge of the Large Language Models with Software Engineering field.
Please feel free to contact us if you have any related references by Github issue or pull request. 

## 👏 Citation

```bib
@article{zhang2023survey,
  title={A Survey on Large Language Models for Software Engineering},
  author={Zhang, Quanjun and Fang, Chunrong and Xie, Yang and Zhang, Yaxin and Yang, Yun and Sun, Weisong and Yu, Shengcheng and Chen, Zhenyu},
  journal={arXiv preprint arXiv:2312.15223},
  year={2023}
}
```

## 📖 Contents
- [👏 Citation](#-citation)
- [📖 Contents](#-contents)
- [🤖RQ1](#rq1)
  - [Encoder-only](#encoder-only)
  - [Encoder-Decoder](#encoder-decoder)
  - [Decoder-only](#decoder-only)
- [💻RQ2](#rq2)
  - [📋Software Requirements \& Design](#software-requirements--design)
    - [Ambiguity detection](#ambiguity-detection)
    - [Class Diagram Derivation](#class-diagram-derivation)
    - [GUI Layouts](#gui-layouts)
    - [Requirement Classification](#requirement-classification)
    - [Requirement Completeness Detection](#requirement-completeness-detection)
    - [Requirement Elicitation](#requirement-elicitation)
    - [Requirement Engineering](#requirement-engineering)
    - [Requirement Prioritization](#requirement-prioritization)
    - [Requirement Summarization](#requirement-summarization)
    - [Requirement Traceability](#requirement-traceability)
    - [Requirements Quality Assurance](#requirements-quality-assurance)
    - [Software Modeling](#software-modeling)
    - [Specification Generation](#specification-generation)
    - [Specifications Repair](#specifications-repair)
    - [Use Case Generation](#use-case-generation)
  - [🛠️Software Development](#️software-development)
    - [API Documentation Smells](#api-documentation-smells)
    - [API Inference](#api-inference)
    - [API recommendation](#api-recommendation)
    - [Code Comment Completion](#code-comment-completion)
    - [Code Completion](#code-completion)
    - [Code Compression](#code-compression)
    - [Code Editing](#code-editing)
    - [Code Generation](#code-generation)
    - [Code Representation](#code-representation)
    - [Code Search](#code-search)
    - [Code Summarization](#code-summarization)
    - [Code Translation](#code-translation)
    - [Code Understanding](#code-understanding)
    - [Continuous Development Optimization](#continuous-development-optimization)
    - [Data Augmentation](#data-augmentation)
    - [Identifier Normalization](#identifier-normalization)
    - [Microservice Recommendation](#microservice-recommendation)
    - [Neural Architecture search](#neural-architecture-search)
    - [Program Synthesis](#program-synthesis)
    - [SO Post Title Generation](#so-post-title-generation)
    - [Type Inference](#type-inference)
    - [Unified Development](#unified-development)
    - [Code recommendation](#code-recommendation)
    - [Control flow graph generation](#control-flow-graph-generation)
    - [Data analysis](#data-analysis)
    - [Method name generation](#method-name-generation)
    - [Project Planning](#project-planning)
    - [SO Question Answering](#so-question-answering)
  - [🧪Software Testing](#software-testing)
    - [Formal verification](#formal-verification)
    - [Invariant Prediction](#invariant-prediction)
    - [proof generation](#proof-generation)
    - [Resource leak detection](#resource-leak-detection)
    - [taint analysis](#taint-analysis)
    - [Actionable Warning Identification](#actionable-warning-identification)
    - [Adversarial Attack](#adversarial-attack)
    - [API Misuse Detection](#api-misuse-detection)
    - [API Testing](#api-testing)
    - [Assertion Generation](#assertion-generation)
    - [Binary Code Similarity Detection](#binary-code-similarity-detection)
    - [Code Execution](#code-execution)
    - [Decompilation](#decompilation)
    - [Failure-Inducing Testing](#failure-inducing-testing)
    - [Fault Localization](#fault-localization)
    - [Fuzzing](#fuzzing)
    - [GUI Testing](#gui-testing)
    - [Indirect Call Analysis](#indirect-call-analysis)
    - [Mutation Testing](#mutation-testing)
    - [NLP Testing](#nlp-testing)
    - [Penetration Testing](#penetration-testing)
    - [Program Analysis](#program-analysis)
    - [Program Reduction](#program-reduction)
    - [Property-based Testing](#property-based-testing)
    - [Simulation Testing](#simulation-testing)
    - [Static Analysis](#static-analysis)
    - [Static Warning Validating](#static-warning-validating)
    - [Test Generation](#test-generation)
    - [Test Suite Minimization](#test-suite-minimization)
    - [Vulnerability Detection](#vulnerability-detection)
    - [Vulnerable Dependency Alert Detection](#vulnerable-dependency-alert-detection)
    - [Theorem Proving](#theorem-proving)
  - [📱Software Maintenance](#software-maintenance)
    - [Android permissions](#android-permissions)
    - [APP Review Analysis](#app-review-analysis)
    - [Bug Report Detection](#bug-report-detection)
    - [Bug Reproduction](#bug-reproduction)
    - [Bug Triaging](#bug-triaging)
    - [Code Clone Detection](#code-clone-detection)
    - [Code Coverage Prediction](#code-coverage-prediction)
    - [Code Evolution](#code-evolution)
    - [Code Porting](#code-porting)
    - [Code Refactoring](#code-refactoring)
    - [Code Review](#code-review)
    - [Code Smells](#code-smells)
    - [Commit Message Generation](#commit-message-generation)
    - [Compiler Optimization](#compiler-optimization)
    - [Debugging](#debugging)
    - [Exception Handling Recommendation](#exception-handling-recommendation)
    - [Flaky Test Prediction](#flaky-test-prediction)
    - [Incident Management](#incident-management)
    - [Issue Labeling](#issue-labeling)
    - [Log Analysis](#log-analysis)
    - [Log Anomaly Detection](#log-anomaly-detection)
    - [Malware Tracker](#malware-tracker)
    - [Mobile app crash detection](#mobile-app-crash-detection)
    - [Outage Understanding](#outage-understanding)
    - [Patch Correctness Assessment](#patch-correctness-assessment)
    - [Privacy Policy](#privacy-policy)
    - [Program Repair](#program-repair)
    - [Report Severity Prediction](#report-severity-prediction)
    - [Sentiment analysis](#sentiment-analysis)
    - [Tag Recommendation](#tag-recommendation)
    - [Technical Debt Management](#technical-debt-management)
    - [Test Update](#test-update)
    - [Traceability Link Recovery](#traceability-link-recovery)
    - [Vulnerability Repair](#vulnerability-repair)
    - [Code Clone Detection](#code-clone-detection-1)
  - [📈Software Management](#software-management)
    - [Developers' Behavior Analysis](#developers-behavior-analysis)
    - [Effort estimation](#effort-estimation)
    - [Software Repository Mining](#software-repository-mining)
    - [Software tool configuration](#software-tool-configuration)
- [🧩RQ3](#rq3)
  - [📊Benchmark](#benchmark)
  - [🗜️Compressing\&Distillation](#️compressingdistillation)
  - [📚Education](#education)
  - [🧮Empirical](#empirical)
  - [🎛️Tuning](#️tuning)



## 🤖RQ1

### 🔢Encoder-only

1. CuBERT: Learning and evaluating contextual embedding of source code [2020-ICML] [GitHub](https://github.com/google-research/google-research/tree/master/cubert) 
2. CodeBERT: CodeBERT: A Pre-Trained Model for Programming and Natural Languages [2020-EMNLP] [GitHub](https://github.com/microsoft/CodeBERT)  
3. GraphCodeBERT: GraphCodeBERT: Pre-training Code Representations with Data Flow [2021-ICLR] [GitHub](https://github.com/microsoft/CodeBERT)  
4. SOBertBase: Stack Over-Flowing with Results: The Case for Domain-Specific Pre-Training Over One-Size-Fits-All Models [2023-arXiv]  
5. CodeSage: Code Representation Learning At Scale [2024-ICLR] [GitHub](https://github.com/amazon-science/CodeSage)  
6. CoLSBERT: Scaling Laws Behind Code Understanding Model [2024-arXiv] [GitHub](https://github.com/stanford-futuredata/ColBERT)  

### 🔀Encoder-Decoder

1. PyMT5: PyMT5 multi-mode translation of natural language and Python code with transformers [2020-EMNLP] [GitHub](https://github.com/devcartel/pymt5)  
2. CodeT5: CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation [2021-EMNLP] [GitHub](https://github.com/salesforce/CodeT5)  
3. PLBART: Unified Pre-training for Program Understanding and Generation [2021-NAACL] [GitHub](https://github.com/wasiahmad/PLBART)  
4. T5-Learning: Studying the usage of text-to-text transfer transformer to support code-related tasks [2021-ICSE] [GitHub](https://github.com/antonio-mastropaolo/T5-learning-ICSE_2021)  
5. CodeRL: CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning [2022-NeurIPS] [GitHub](https://github.com/salesforce/CodeRL)  
6. CoditT5: CoditT5: Pretraining for Source Code and Natural Language Editing [2022-ASE] [GitHub](https://github.com/engineeringsoftware/coditt5)  
7. JuPyT5: Training and Evaluating a Jupyter Notebook Data Science Assistant [2022-arXiv] [GitHub](https://github.com/microsoft/DataScienceProblems)  
8. SPT-Code: Sequence-to-Sequence Pre-Training for Learning Source Code Representations [2022-ICSE] [GitHub](https://github.com/NougatCA/SPT-Code)  
9. UnixCoder: UniXcoder: Unified Cross-Modal Pre-training for Code Representation [2022-ACL] [GitHub](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder)  
10. AlphaCode: Competition-Level Code Generation with AlphaCode [2022-Science]  
11. ERNIE-Code: ERNIE-Code: Beyond English-Centric Cross-lingual Pretraining for Programming Languages [2023-ACL] [GitHub](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-code)  
12. CodeT5+: CodeT5+ Open Code Large Language Models for Code Understanding and Generation [2023-EMNLP] [GitHub](https://github.com/salesforce/CodeT5/tree/main/CodeT5%2B)  
13. PPOCoder: Execution-based Code Generation using Deep Reinforcement Learning [2023-TMLR] [GitHub](https://github.com/reddy-lab-code-research/PPOCoder)  
14. RLTF: RLTF: Reinforcement Learning from Unit Test Feedback [2023-TMLR] [GitHub](https://github.com/Zyq-scut/RLTF)  
15. CCT5: CCT5: A Code-Change-Oriented Pre-Trained Model [2023-FSE] [GitHub](https://github.com/Ringbo/CCT5)  
16. B-Coder: B-Coder: Value-Based Deep Reinforcement Learning for Program Synthesis [2024-ICLR]  
17. AST-T5: AST-T5: Structure-Aware Pretraining for Code Generation and Understanding [2024-ICML] [GitHub](https://github.com/gonglinyuan/ast_t5)  
18. GrammarT5: GrammarT5: Grammar-Integrated Pretrained Encoder-Decoder Neural Model for Code [2024-ICSE] [GitHub](https://github.com/pkuzqh/GrammarT5)  

### 🧩Decoder-only

1. GPT-C: IntelliCode compose code generation using transformer [2020-FSE]  
2. Codex: Evaluating large language models trained on code [2021-arXiv]  
3. CodeGPT: CodeXGLUE A Machine Learning Benchmark Dataset for Code Understanding and Generation [2021-NeurIPS] [GitHub](https://github.com/microsoft/CodeXGLUE)  
4. PaLM-Coder: PaLM Scaling Language Modeling with Pathways [2022-JMLR]  
5. PanGu-Coder: PanGu-Coder Program Synthesis with Function-Level Language Modeling [2022-arXiv]  
6. PolyCoder: A Systematic Evaluation of Large Language Models of Code [2022-ICLR] [GitHub](https://github.com/VHellendoorn/Code-LMs)  
7. PyCodeGPT: CERT Continual Pre-Training on Sketches for Library-Oriented Code Generation [2022-IJCAI] [GitHub](https://github.com/microsoft/pycodegpt)  
8. BLOOM: BLOOM: A 176B-Parameter Open-Access Multilingual Language Model [2022-arXiv] [GitHub](https://huggingface.co/bigscience/bloom)  
9. CodeShell: CodeShell Technical Report [2023-arXiv]  
10. PanGu-Coder2: PanGu-Coder2: LLM with Reinforcement Learning [2023-arXiv]  
11. Code Llama: Code llama: Open foundation models for code [2023-arXiv] [GitHub](https://github.com/facebookresearch/codellama)  
12. CodeFuse: CodeFuse-13B: A Pretrained Multi-lingual Code Large Language Model [2023-ICSE] [GitHub](https://github.com/codefuse-ai)  
13. CodeGen: CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis [2023-ICLR] [GitHub](https://github.com/salesforce/CodeGen)  
14. CodeGen2: CodeGen2 Lessons for Training LLMs on Programming and Natural Languages [2023-ICLR] [GitHub](https://github.com/salesforce/CodeGen2)  
15. InCoder: InCoder: A Generative Model for Code Infilling and Synthesis [2023-ICLR] [GitHub](https://sites.google.com/view/incoder-code-models)  
16. SantaCoder: SantaCoder don’t reach for the stars! [2023-ICLR] [GitHub](https://huggingface.co/bigcode/santacoder)  
17. StarCoder: StarCoder may the source be with you [2023-TMLR] [GitHub](https://github.com/bigcode-project/starcoder)  
18. CodeGeeX: CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X [2024-KDD] [GitHub](https://github.com/THUDM/CodeGeeX)  
19. Lemur: Lemur: Harmonizing Natural Language and Code for Language Agents [2024-ICLR] [GitHub](https://github.com/OpenLemur/Lemur)  
20. Magicoder: Magicoder: Empowering Code Generation with OSS-Instruct [2024-ICML] [GitHub](https://github.com/ise-uiuc/magicoder)  
21. OctoCoder: Octopack: Instruction tuning code large language models [2024-ICLR] [GitHub](https://huggingface.co/bigcode/octocoder)  
22. WizardCoder: WizardCoder: Empowering Code Large Language Models with Evol-Instruct [2024-ICLR] [GitHub](https://github.com/nlpxucan/WizardLM)  
23. AlchemistCoder: AlchemistCoder: Harmonizing and Eliciting Code Capability by Hindsight Tuning on Multi-source Data [2024-arXiv] [GitHub](https://github.com/InternLM/AlchemistCoder)  
24. AutoCoder: AutoCoder: Enhancing Code Large Language Model with AIEV-Instruct [2024-arXiv] [GitHub](https://github.com/bin123apple/AutoCoder)  
25. CodeGemma: CodeGemma: Open Code Models Based on Gemma [2024-arXiv] [GitHub](https://huggingface.co/blog/codegemma)  
26. DeepSeek-Coder: DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence [2024-arXiv] [GitHub](https://github.com/deepseek-ai/DeepSeek-Coder)  
27. DeepSeek-Coder-V2: DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence [2024-arXiv] [GitHub](https://github.com/deepseek-ai/DeepSeek-Coder-V2)  
28. DolphCoder: DolphCoder: Echo-Locating Code Large Language Models with Diverse and Multi-Objective Instruction Tuning [2024-ACL] [GitHub](https://github.com/pris-nlp/DolphCoder)  
29. Granite: Granite Code Models: A Family of Open Foundation Models for Code Intelligence [2024-arXiv] [GitHub](https://github.com/ibm-granite/granite-code-models)  
30. InverseCoder: InverseCoder: Unleashing the Power of Instruction-Tuned Code LLMs with Inverse-Instruct [2024-arXiv] [GitHub](https://github.com/wyt2000/InverseCoder)  
31. NT-Java: Narrow Transformer: Starcoder-Based Java-LM For Desktop [2024-arXiv] [GitHub](https://huggingface.co/infosys/NT-Java-1.1B)  
32. StarCoder2: StarCoder 2 and The Stack v2: The Next Generation [2024-arXiv] [GitHub](https://github.com/bigcode-project/starcoder2)  
33. StepCoder: [2024-ACL] [GitHub](https://github.com/Ablustrund/APPS_Plus)  
34. UniCoder: UniCoder: Scaling Code Large Language Model via Universal Code [2024-ACL] [GitHub](https://github.com/microsoft/Unicoder)  
35. WaveCoder: WaveCoder: Widespread And Versatile Enhanced Code LLM [2024-ACL] [GitHub](https://github.com/microsoft/WaveCoder)  
36. XFT: XFT: Unlocking the Power of Code Instruction Tuning by Simply Merging Upcycled Mixture-of-Experts [2024-ACL] [GitHub](https://github.com/ise-uiuc/xft)  

    

## 💻RQ2

### 📋Software Requirements \& Design

#### Ambiguity detection

1. Automated Handling of Anaphoric Ambiguity in Requirements: A Multi-Solution Study [2022-ICSE]
2. Automated requirement contradiction detection through formal logic and LLMs [2024-AUSE]
3. Identification of intra-domain ambiguity using transformer-based machine learning [2022-ICSE@NLBSE]
4. TABASCO: A transformer based contextualization toolkit [2022-SCP]
5. ChatGPT: A Study on its Utility for Ubiquitous Software Engineering Tasks [2023-arXiv]

#### Class Diagram Derivation

1. LLM-based Class Diagram Derivation from User Stories with Chain-of-Thought Promptings [2024-COMPSAC]
2. A hybrid approach to extract conceptual diagram from software requirements [2024-SCP]

#### GUI Layouts

1. Data-driven prototyping via natural-language-based GUI retrieval [2023-AUSE]
2. Evaluating a Large Language Model on Searching for GUI Layouts [2023-EICS]

#### Requirement Classification

1. Which AI Technique Is Better to Classify Requirements? An Experiment with SVM, LSTM, and ChatGPT [2023-arXiv]
2. Improving Requirements Classification Models Based on Explainable Requirements Concerns [2023-REW]
3. NoRBERT: Transfer Learning for Requirements Classification [2020-RE]
4. Non Functional Requirements Identification and Classification Using Transfer Learning Model [2023-IEEE Access]
5. PRCBERT: Prompt Learning for Requirement Classification using BERT-based Pretrained Language Models [2022-ASE]
6. Pre-trained Model-based NFR Classification: Overcoming Limited Data Challenges [2023-IEEE Access]
7. BERT-Based Approach for Greening Software Requirements Engineering Through Non-Functional Requirements [2023-IEEE Access]

#### Requirement Completeness Detection

1. Improving requirements completeness: Automated assistance through large language models [2024-RE]


#### Requirement Elicitation

1. Combining Prompts with Examples to Enhance LLM-Based Requirement Elicitation [2024-COMPSAC]


#### Requirement Engineering

1. Advancing Requirements Engineering through Generative AI: Assessing the Role of LLMs [2024-Generative AI]
2. Lessons from the Use of Natural Language Inference (NLI) in Requirements Engineering Tasks [2024-arXiv]
3. Enhancing Legal Compliance and Regulation Analysis with Large Language Models [2024-arXiv]
4. MARE: Multi-Agents Collaboration Framework for Requirements Engineering [2024-arXiv]
5. Multilingual Crowd-Based Requirements Engineering Using Large Language Models [2024-SBES]
6. Requirements Engineering using Generative AI: Prompts and Prompting Patterns [2024-Generative AI]
7. From Specifications to Prompts: On the Future of Generative LLMs in Requirements Engineering [2024-IEEE Software]

#### Requirement Prioritization

1. Prioritizing Software Requirements Using Large Language Models [2024-arXiv]


#### Requirement Summarization

1. A Transformer-based Approach for Abstractive Summarization of Requirements from Obligations in Software Engineering Contracts [2023-RE]


#### Requirement Traceability

1. Natural Language Processing for Requirements Traceability [2024-arXiv]
2. Traceability Transformed: Generating More Accurate Links with Pre-Trained BERT Models [2021-ICSE]

#### Requirements Quality Assurance

1. Leveraging LLMs for the Quality Assurance of Software Requirements [2024-RE]
2. Leveraging Transformer-based Language Models to Automate Requirements Satisfaction Assessment [2023-arXiv]
3. Supporting High-Level to Low-Level Requirements Coverage Reviewing with Large Language Models [2024-MSR]
4. ChatGPT as a tool for User Story Quality Evaluation: Trustworthy Out of the Box? [2023-XP]

#### Software Modeling

1. Towards using Few-Shot Prompt Learning for Automating Model Completion [2022-ICSE@NIER]
2. Model Generation from Requirements with LLMs: an Exploratory Study [2024-arXiv]
3. Leveraging Large Language Models for Software Model Completion: Results from Industrial and Public Datasets [2024-arXiv]
4. How LLMs Aid in UML Modeling: An Exploratory Study with Novice Analysts [2024-arXiv]
5. Natural Language Processing-based Requirements Modeling: A Case Study on Problem Frames [2023-APSEC]

#### Specification Generation

1. SpecGen: Automated Generation of Formal Program Specifications via Large Language Models [2024-arXiv]
2. Large Language Models Based Automatic Synthesis of Software Specifications [2023-arXiv]
3. Impact of Large Language Models on Generating Software Specifications [2023-arXiv]

#### Specifications Repair

1. Automated Repair of Declarative Software Specifications in the Era of Large Language Models [2023-arXiv]


#### Use Case Generation

1. Experimenting a New Programming Practice with LLMs [2024-arXiv]


### 🛠️Software Development

#### API Documentation Smells

Automatic Detection of Five API Documentation Smells: Practitioners’ Perspectives [2021-SANER]

#### API Inference

1. Adaptive Intellect Unleashed: The Feasibility of Knowledge Transfer in Large Language Models [2023-arXiv]
2. Gorilla: Large language model connected with massive APIs [2023-arXiv]
3. Measuring and Mitigating Constraint Violations of In-Context Learning for Utterance-to-API Semantic Parsing [2023-arXiv]
4. Pop Quiz! Do Pre-trained Code Models Possess Knowledge of Correct API Names? [2023-arXiv]

#### API recommendation

1. APIGen: Generative API Method Recommendation [2024-SANER]
2. Let's Chat to Find the APIs: Connecting Human, LLM and Knowledge Graph through AI Chain [2023-ASE]
3. PTM-APIRec: Leveraging Pre-trained Models of Source Code in API Recommendation [2023-TOSEM]
4. CLEAR: Contrastive Learning for API Recommendation [2022-ICSE]
5. Automatic recognizing relevant fragments of APIs using API references [2024-AUSE]
6. ToolCoder: Teach Code Generation Models to use API search tools [2023-arXiv]

#### Code Comment Completion

1. APIGen: Generative API Method Recommendation [2024-SANER]
2. Let's Chat to Find the APIs: Connecting Human, LLM and Knowledge Graph through AI Chain [2023-ASE]
3. PTM-APIRec: Leveraging Pre-trained Models of Source Code in API Recommendation [2023-TOSEM]
4. CLEAR: Contrastive Learning for API Recommendation [2022-ICSE]
5. Automatic recognizing relevant fragments of APIs using API references [2024-AUSE]
6. ToolCoder: Teach Code Generation Models to use API search tools [2023-arXiv]

#### Code Completion

1. Towards Efficient Fine-tuning of Pre-trained Code Models: An Experimental Study and Beyond [2023-ISSTA]
2. Dataflow-Guided Retrieval Augmentation for Repository-Level Code Completion [2024-ACL]
3. An Empirical Study on the Usage of BERT Models for Code Completion [2021-MSR]
4. An Empirical Study on the Usage of Transformer Models for Code Completion [2021-TSE]
5. R2C2-Coder: Enhancing and Benchmarking Real-world Repository-level Code Completion Abilities of Code Large Language Models [2024-arXiv]
6. A Static Evaluation of Code Completion by Large Language Models [2023-ACL]
7. CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion [2023-NeurIPS]
8. Large Language Models of Code Fail at Completing Code with Potential Bugs [2024-NeurIPS]
9. Piloting Copilot and Codex: Hot Temperature, Cold Prompts, or Black Magic? [2022-arXiv]
10. De-Hallucinator: Iterative Grounding for LLM-Based Code Completion [2024-arXiv]
11. Evaluation of LLMs on Syntax-Aware Code Fill-in-the-Middle Tasks [2024-ICML]
12. Codefill: Multi-token code completion by jointly learning from structure and naming sequences [2022-ICSE]
13. ZS4C: Zero-Shot Synthesis of Compilable Code for Incomplete Code Snippets using ChatGPT [2024-arXiv]
14. Automatic detection and analysis of technical debts in peer-review documentation of r packages [2022-SANER]
15. Toward less hidden cost of code completion with acceptance and ranking models [2021-ICSME]
16. Enhancing LLM-Based Coding Tools through Native Integration of IDE-Derived Static Context [2024-arXiv]
17. RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems [2023-ICLR]
18. GraphCoder: Enhancing Repository-Level Code Completion via Code Context Graph-based Retrieval and Language Model [2025-arXiv]
19. STALL+: Boosting LLM-based Repository-level Code Completion with Static Analysis [2024-arXiv]
20. CCTEST: Testing and Repairing Code Completion Systems [2023-ICSE]
21. Contextual API Completion for Unseen Repositories Using LLMs [2024-arXiv]
22. Learning Deep Semantics for Test Completion [2023-ICSE]
23. Evaluating and improving transformers pre-trained on asts for code completion [2023-SANER]
24. RepoHyper: Better Context Retrieval Is All You Need for Repository-Level Code Completion [2024-arXiv]
25. Making the most of small Software Engineering datasets with modern machine learning [2021-TSE]
26. From copilot to pilot: Towards AI supported software development [2023-arXiv]
27. When Neural Code Completion Models Size up the Situation: Attaining Cheaper and Faster Completion through Dynamic Model Inference [2024-ICSE]
28. Prompt-based Code Completion via Multi-Retrieval Augmented Generation [2024-arXiv]
29. Domain Adaptive Code Completion via Language Models and Decoupled Domain Databases [2023-ASE]
30. Enriching Source Code with Contextual Data for Code Completion Models: An Empirical Study [2023-MSR]
31. RLCoder: Reinforcement Learning for Repository-Level Code Completion [2024-ICSE]
32. Repoformer: Selective Retrieval for Repository-Level Code Completion [2024-ICML]
33. A systematic evaluation of large language models of code [2022-PLDI]
34. Hierarchical Context Pruning: Optimizing Real-World Code Completion with Repository-Level Pretrained Code LLMs [2024-arXiv]
35. LLM-Cloud Complete: Leveraging Cloud Computing for Efficient Large Language Model-based Code Completion [2024-JAIGS]

#### Code Compression

1. Semantic Compression With Large Language Models [2023-SNAMS]
2. On the validity of pre-trained transformers for natural language processing in the software engineering domain [2022-TSE]

#### Code Editing

1. Unprecedented Code Change Automation: The Fusion of LLMs and Transformation by Example [2024-FSE]
2. GrACE: Generation using Associated Code Edits [2023-FSE]
3. CodeEditor: Learning to Edit Source Code with Pre-trained Models [2023-TOSEM]
4. Automated Code Editing with Search-Generate-Modify [2024-ICSE]
5. Coffee: Boost Your Code LLMs by Fixing Bugs with Feedback [2023-arXiv]

#### Code Generation

1. Towards Efficient Fine-tuning of Pre-trained Code Models: An Experimental Study and Beyond [2023-ISSTA]
2. Dataflow-Guided Retrieval Augmentation for Repository-Level Code Completion [2024-ACL]
3. An Empirical Study on the Usage of BERT Models for Code Completion [2021-MSR]
4. An Empirical Study on the Usage of Transformer Models for Code Completion [2021-TSE]
5. R2C2-Coder: Enhancing and Benchmarking Real-world Repository-level Code Completion Abilities of Code Large Language Models [2024-arXiv]
6. A Static Evaluation of Code Completion by Large Language Models [2023-ACL]
7. CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion [2023-NeurIPS]
8. Large Language Models of Code Fail at Completing Code with Potential Bugs [2024-NeurIPS]
9. Piloting Copilot and Codex: Hot Temperature, Cold Prompts, or Black Magic? [2022-arXiv]
10. De-Hallucinator: Iterative Grounding for LLM-Based Code Completion [2024-arXiv]
11. Evaluation of LLMs on Syntax-Aware Code Fill-in-the-Middle Tasks [2024-ICML]
12. Codefill: Multi-token code completion by jointly learning from structure and naming sequences [2022-ICSE]
13. ZS4C: Zero-Shot Synthesis of Compilable Code for Incomplete Code Snippets using ChatGPT [2024-arXiv]
14. Automatic detection and analysis of technical debts in peer-review documentation of r packages [2022-SANER]
15. Toward less hidden cost of code completion with acceptance and ranking models [2021-ICSME]
16. Enhancing LLM-Based Coding Tools through Native Integration of IDE-Derived Static Context [2024-arXiv]
17. RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems [2023-ICLR]
18. GraphCoder: Enhancing Repository-Level Code Completion via Code Context Graph-based Retrieval and Language Model [2025-arXiv]
19. STALL+: Boosting LLM-based Repository-level Code Completion with Static Analysis [2024-arXiv]
20. CCTEST: Testing and Repairing Code Completion Systems [2023-ICSE]
21. Contextual API Completion for Unseen Repositories Using LLMs [2024-arXiv]
22. Learning Deep Semantics for Test Completion [2023-ICSE]
23. Evaluating and improving transformers pre-trained on asts for code completion [2023-SANER]
24. RepoHyper: Better Context Retrieval Is All You Need for Repository-Level Code Completion [2024-arXiv]
25. Making the most of small Software Engineering datasets with modern machine learning [2021-TSE]
26. From copilot to pilot: Towards AI supported software development [2023-arXiv]
27. When Neural Code Completion Models Size up the Situation: Attaining Cheaper and Faster Completion through Dynamic Model Inference [2024-ICSE]
28. Prompt-based Code Completion via Multi-Retrieval Augmented Generation [2024-arXiv]
29. Domain Adaptive Code Completion via Language Models and Decoupled Domain Databases [2023-ASE]
30. Enriching Source Code with Contextual Data for Code Completion Models: An Empirical Study [2023-MSR]
31. RLCoder: Reinforcement Learning for Repository-Level Code Completion [2024-ICSE]
32. Repoformer: Selective Retrieval for Repository-Level Code Completion [2024-ICML]
33. A systematic evaluation of large language models of code [2022-PLDI]
34. Hierarchical Context Pruning: Optimizing Real-World Code Completion with Repository-Level Pretrained Code LLMs [2024-arXiv]
35. LLM-Cloud Complete: Leveraging Cloud Computing for Efficient Large Language Model-based Code Completion [2024-JAIGS]
36. A Closer Look at Different Difficulty Levels Code Generation Abilities of ChatGPT [2023-ASE]
37. Enhancing Code Intelligence Tasks with ChatGPT [2023-arXiv]
38. ExploitGen: Template-augmented exploit code generation based on CodeBERT [2024-JSS]
39. A syntax-guided multi-task learning approach for Turducken-style code generation [2023-EMSE]
40. CoLadder: Supporting Programmers with Hierarchical Code Generation in Multi-Level Abstraction [2023-arXiv]
41. Evaluating the Code Quality of AI-Assisted Code Generation Tools: An Empirical Study on GitHub Copilot, Amazon CodeWhisperer, and ChatGPT [2023-arXiv]
42. CoderEval: A Benchmark of Pragmatic Code Generation with Generative Pre-trained Models [2023-ICSE]
43. CERT: Continual Pre-training on Sketches for Library-oriented Code Generation [2022-IJCAI]
44. When language model meets private library [2022-EMNLP]
45. Private-Library-Oriented Code Generation with Large Language Models [2023-arXiv]
46. Self-taught optimizer (stop): Recursively self-improving code generation [2024-COLM]
47. Coder reviewer reranking for code generation [2023-ICML]
48. Planning with Large Language Models for Code Generation [2023-ICLR]
49. CodeAgent: Enhancing Code Generation with Tool-Integrated Agent Systems for Real-World Repo-level Coding Challenges [2024-ACL]
50. A Lightweight Framework for Adaptive Retrieval In Code Completion With Critique Model [2024-arXiv]
51. Self-Edit: Fault-Aware Code Editor for Code Generation [2023-ACL]
52. Outline, then details: Syntactically guided coarse-to-fine code generation [2023-ICML]
53. Self-Infilling Code Generation [2024-ICML]
54. Can ChatGPT replace StackOverflow? A Study on Robustness and Reliability of Large Language Model Code Generation [2023-arXiv]
55. Can LLM Replace Stack Overflow? A Study on Robustness and Reliability of Large Language Model Code Generation [2024-AAAI]
56. CodeBERTScore: Evaluating Code Generation with Pretrained Models of Code [2023-EMNLP]
57. Hot or Cold? Adaptive Temperature Sampling for Code Generation with Large Language Models [2024-AAAI]
58. Sketch Then Generate: Providing Incremental User Feedback and Guiding LLM Code Generation through Language-Oriented Code Sketches [2024-arXiv]
59. Two Birds with One Stone: Boosting Code Generation and Code Search via a Generative Adversarial Network [2023-OOPSLA]
60. Natural Language to Code: How Far Are We? [2023-FSE]

#### Code Representation

1. Structured Code Representations Enable Data-Efficient Adaptation of Code Language Models [2024-arXiv]
2. API2Vec++: Boosting API Sequence Representation for Malware Detection and Classification [2024-TSE]
3. Representation Learning for Stack Overflow Posts: How Far are We? [2024-TOSEM]
4. VarGAN: Adversarial Learning of Variable Semantic Representations [2024-TSE]
5. ContraBERT: Enhancing Code Pre-trained Models via Contrastive Learning [2023-ICSE]
6. Model-Agnostic Syntactical Information for Pre-Trained Programming Language Models [2023-MSR]

#### Code Search

1. One Adapter for All Programming Languages? Adapter Tuning for Code Search and Summarization [2023-ICSE]
2. Two Birds with One Stone: Boosting Code Generation and Code Search via a Generative Adversarial Network [2023-OOPSLA]
3. An Empirical Study on Code Search Pre-trained Models: Academic Progresses vs. Industry Requirements [2024-Internetware]
4. Rapid: Zero-shot Domain Adaptation for Code Search with Pre-trained Models [2024-TOSEM]
5. LLM Agents Improve Semantic Code Search [2024-arXiv]
6. Generation-Augmented Query Expansion For Code Retrieval [2022-arXiv]
7. MCodeSearcher: Multi-View Contrastive Learning for Code Search [2024-Internetware]
8. Do Pre-trained Language Models Indeed Understand Software Engineering Tasks? [2022-TSE]
9. Rewriting the Code: A Simple Method for Large Language Model Augmented Code Search [2024-ACL]
10. CodeRetriever: A Large Scale Contrastive Pre-Training Method for Code Search [2022-EMNLP]
11. Self-Supervised Query Reformulation for Code Search [2023-FSE]
12. On Contrastive Learning of Semantic Similarity for Code to Code Search [2023-arXiv]
13. On the Effectiveness of Transfer Learning for Code Search [2022-TSE]
14. Cross-modal contrastive learning for code search [2022-ICSME]
15. CoCoSoDa: Effective Contrastive Learning for Code Search [2023-ICSE]
16. Improving code search with multi-modal momentum contrastive learning [2023-ICPC]
17. CodeCSE: A Simple Multilingual Model for Code and Comment Sentence Embeddings [2024-arXiv]
18. You Augment Me: Exploring ChatGPT-based Data Augmentation for Semantic Code Search [2023-ICSME]
19. Natural Language to Code: How Far Are We? [2023-FSE]
20. CCT-Code: Cross-Consistency Training for Multilingual Clone Detection and Code Search [2023-arXiv]

#### Code Summarization

1. Distilled GPT for Source Code Summarization [2024-AUSE]
2. One Adapter for All Programming Languages? Adapter Tuning for Code Search and Summarization [2023-ICSE]
3. Automatic Semantic Augmentation of Language Model Prompts (for Code Summarization) [2024-ICSE]
4. Extending Source Code Pre-Trained Language Models to Summarise Decompiled Binaries [2023-SANER]
5. Exploring Distributional Shifts in Large Language Models for Code Analysis [2023-EMNLP]
6. On the transferability of pre-trained language models for low-resource programming languages [2022-ICPC]
7. A Comparative Analysis of Large Language Models for Code Documentation Generation [2024-arXiv]
8. Dialog summarization for software collaborative platform via tuning pre-trained models [2023-JSS]
9. ESALE: Enhancing Code-Summary Alignment Learning for Source Code Summarization [2024-TSE]
10. Constructing effective in-context demonstration for code intelligence tasks: An empirical study [2023-arXiv]
11. Large Language Models are Few-Shot Summarizers: Multi-Intent Comment Generation via In-Context Learning [2024-ICSE]
12. Assemble foundation models for automatic code summarization [2022-SANER]
13. Analyzing the performance of large language models on code summarization [2024-arXiv]
14. Binary code summarization: Benchmarking chatgpt/gpt-4 and other large language models [2023-arXiv]
15. Binary Code Summarization: Benchmarking ChatGPT/GPT-4 and Other Large Language Models [2023-arXiv]
16. SimLLM: Measuring Semantic Similarity in Code Summaries Using a Large Language Model-Based Approach [2024-FSE]
17. Identifying Inaccurate Descriptions in LLM-generated Code Comments via Test Execution [2024-arXiv]
18. Code Summarization without Direct Access to Code - Towards Exploring Federated LLMs for Software Engineering [2024-EASE]
19. Cross-Modal Retrieval-enhanced code Summarization based on joint learning for retrieval and generation [2024-IST]
20. Do Machines and Humans Focus on Similar Code? Exploring Explainability of Large Language Models in Code Summarization [2024-ICPC]
21. MALSIGHT: Exploring Malicious Source Code and Benign Pseudocode for Iterative Binary Malware Summarization [2024-arXiv]
22. CSA-Trans: Code Structure Aware Transformer for AST [2024-arXiv]
23. Exploring the Efficacy of Large Language Models (GPT-4) in Binary Reverse Engineering [2024-arXiv]
24. DocuMint: Docstring Generation for Python using Small Language Models [2024-arXiv]
25. Achieving High-Level Software Component Summarization via Hierarchical Chain-of-Thought Prompting and Static Code Analysis [2023-ICoDSE]
26. Multilingual Adapter-based Knowledge Aggregation on Code Summarization for Low-Resource Languages [2023-arXiv]
27. Analysis of ChatGPT on Source Code [2023-arXiv]
28. Bash comment generation via data augmentation and semantic-aware CodeBERT [2024-AUSE]
29. SoTaNa: The Open-Source Software Development Assistant [2023-arXiv]
30. Natural Language Outlines for Code: Literate Programming in the LLM Era [2024-arXiv]
31. Semantic Similarity Loss for Neural Source Code Summarization [2023-JSEP]
32. Context-aware Code Summary Generation [2024-arXiv]
33. Automatic Code Summarization via ChatGPT: How Far Are We? [2023-arXiv]
34. A Prompt Learning Framework for Source Code Summarization [2023-TOSEM]
35. Source Code Summarization in the Era of Large Language Models [2024-ICSE]
36. Automatic Code Summarization via ChatGPT- How Far Are We? [2023-arXiv]
37. Large Language Models for Code Summarization [2024-arXiv]
38. Enhancing Trust in LLM-Generated Code Summaries with Calibrated Confidence Scores [2024-arXiv]
39. Generating Variable Explanations via Zero-shot Prompt Learning [2023-ASE]
40. Natural Is The Best: Model-Agnostic Code Simplification for Pre-trained Large Language Models [2024-arXiv]
41. SparseCoder: Identifier-Aware Sparse Transformer for File-Level Code Summarization [2024-SANER]
42. Automatic smart contract comment generation via large language models and in-context learning [2024-IST]
43. Prompt Engineering or Fine Tuning: An Empirical Assessment of Large Language Models in Automated Software Engineering Tasks [2023-arXiv]

#### Code Translation

1. Learning Transfers over Several Programming Languages [2023-arXiv]
2. Enhancing Code Translation in Language Models with Few-Shot Learning via Retrieval-Augmented Generation [2024-arXiv]
3. Codetf: One-stop transformer library for state-of-the-art code llm [2023-arXiv]
4. LASSI: An LLM-based Automated Self-Correcting Pipeline for Translating Parallel Scientific Codes [2024-arXiv]
5. Towards Translating Real-World Code with LLMs: A Study of Translating to Rust [2024-arXiv]
6. Program Translation via Code Distillation [2023-EMNLP]
7. CoTran: An LLM-based Code Translator using Reinforcement Learning with Feedback from Compiler and Symbolic Execution [2023-arXiv]
8. Few-shot code translation via task-adapted prompt learning [2024-JSS]
9. Exploring the Impact of the Output Format on the Evaluation of Large Language Models for Code Translation [2024-Forge]
10. SpecTra: Enhancing the Code Translation Ability of Language Models by Generating Multi-Modal Specifications [2024-arXiv]
11. SteloCoder: a Decoder-Only LLM for Multi-Language to Python Code Translation [2023-arXiv]
12. Lost in Translation: A Study of Bugs Introduced by Large Language Models while Translating Code [2024-ICSE]
13. Understanding the effectiveness of large language models in code translation [2023-ICSE]
14. SUT: Active Defects Probing for Transcompiler Models [2023-EMNLP]
15. Explain-then-Translate: An Analysis on Improving Program Translation with Self-generated Explanations [2023-EMNLP]
16. TransMap: Pinpointing Mistakes in Neural Code Translation [2023-FSE]
17. An interpretable error correction method for enhancing code-to-code translation [2024-ICLR]
18. Codetransocean: A comprehensive multilingual benchmark for code translation [2023-arXiv]
19. Assessing and Improving Syntactic Adversarial Robustness of Pre-trained Models for Code Translation [2023-ICSE]
20. Exploring and unleashing the power of large language models in automated code translation [2024-FSE]
21. VERT: Verified Equivalent Rust Transpilation with Few-Shot Learning [2024-arXiv]
22. Rectifier: Code Translation with Corrector via LLMs [2024-arXiv]
23. Multilingual Code Snippets Training for Program Translation [2022-AAAI]
24. On the Evaluation of Neural Code Translation: Taxonomy and Benchmark [2023-ASE]

#### Code Understanding

1. BinBert: Binary Code Understanding with a Fine-tunable and Execution-aware Transformer [2024-TDSC]
2. SEMCODER: Training Code Language Models with Comprehensive Semantics [2024-arXiv]
3. PAC Prediction Sets for Large Language Models of Code [2023-ICML]
4. The Scope of ChatGPT in Software Engineering: A Thorough Investigation [2023-arXiv]
5. ART: Automatic multi-step reasoning and tool-use for large language models [2023-arXiv]
6. Better Context Makes Better Code Language Models: A Case Study on Function Call Argument Completion [2023-AAAI]
7. Benchmarking Language Models for Code Syntax Understanding [2022-EMNLP]
8. ShellGPT: Generative Pre-trained Transformer Model for Shell Language Understanding [2023-ISSRE]
9. Language Agnostic Code Embeddings [2024-NAACL]
10. Understanding Programs by Exploiting (Fuzzing) Test Cases [2023-arXiv]
11. Can Machines Read Coding Manuals Yet? -- A Benchmark for Building Better Language Models for Code Understanding [2022-AAAI]
12. Using an LLM to Help With Code Understanding [2024-ICSE]

#### Continuous Development Optimization

1. Optimizing Continuous Development By Detecting and Preventing Unnecessary Content Generation [2023-ASE]

#### Data Augmentation

1. A Transformer-based Approach for Augmenting Software Engineering Chatbots Datasets [2024-ESEM]
2. PERFGEN: A Synthesis and Evaluation Framework for Performance Data using Generative AI [2024-COMPSAC]

#### Identifier Normalization

1. BEQAIN: An Effective and Efficient Identifier Normalization Approach with BERT and the Question Answering System [2022-TSE]

#### Microservice Recommendation

1. MicroRec: Leveraging Large Language Models for Microservice Recommendation [2024-MSR]

#### Neural Architecture search

1. LLMatic: Neural Architecture Search via Large Language Models and Quality-Diversity Optimization [2023-GECOO]

#### Program Synthesis

1. Program synthesis with large language models [2021-arXiv]
2. HYSYNTH: Context-Free LLM Approximation for Guiding Program Synthesis [2024-arXiv]
3. Natural Language Commanding via Program Synthesis [2023-arXiv]
4. Function-constrained Program Synthesis [2023-arXiv]
5. Jigsaw: Large Language Models meet Program Synthesis [2022-ICSE]
6. Less is More: Summary of Long Instructions is Better for Program Synthesis [2022-EMNLP]
7. Guiding enumerative program synthesis with large language models [2024-CAV]
8. Fully Autonomous Programming with Large Language Models [2023-GECCO]
9. Exploring the Robustness of Large Language Models for Solving Programming Problems [2023-arXiv]
10. Evaluating ChatGPT and GPT-4 for Visual Programming [2023-ICER]
11. Enhancing Program Synthesis with Large Language Models Using Many-Objective Grammar-Guided Genetic Programming [2024-Algorithms]
12. Synergistic Utilization of LLMs for Program Synthesis [2024-GECCO]
13. Generating Data for Symbolic Language with Large Language Models [2023-EMNLP]

#### SO Post Title Generation

1. Good things come in three: Generating SO Post Titles with Pre-Trained Models, Self Improvement and Post Ranking [2024-ESEM]
2. Automatic bi-modal question title generation for Stack Overflow with prompt learning [2024-EMSE]

#### Type Inference

1. Learning to Predict User-Defined Types [2022-TSE]

#### Unified Development

1. Chatdev: Communicative agents for software development [2023-ACL]

#### Code recommendation

1. Improving code example recommendations on informal documentation using bert and query-aware lsh: A comparative study [2023-arXiv]
2. GraphPyRec: A novel graph-based approach for fine-grained Python code recommendation [2024-SCP]

#### Control flow graph generation

1. AI Chain on Large Language Model for Unsupervised Control Flow Graph Generation for Statically-Typed Partial Code [2023-arXiv]

#### Data analysis

1. Is GPT-4 a Good Data Analyst? [2023-EMNLP]

#### Method name generation

1. Automating Method Naming with Context-Aware Prompt-Tuning [2023-ICPC]

#### Project Planning

1. AutoScrum: Automating Project Planning Using Large Language Models [2023-arXiv]

#### SO Question Answering

1. Time to separate from StackOverflow and match with ChatGPT for encryption [2024-JSS]
2. Is Stack Overflow Obsolete? An Empirical Study of the Characteristics of ChatGPT Answers to Stack Overflow Questions [2024-CHI]

### 🧪Software Testing

#### Formal verification

A New Era in Software Security: Towards Self-Healing Software via Large Language Models and Formal Verification [2023-arXiv]

PropertyGPT: LLM-driven Formal Verification of Smart Contracts through Retrieval-Augmented Property Generation [2024-arXiv]

The FormAI Dataset: Generative AI in Software Security Through the Lens of Formal Verification [2023-PROMISE]

Enchanting Program Specification Synthesis by Large Language Models Using Static Analysis and Program Verification [2024-CAV]

#### Invariant Prediction

Can Large Language Models Reason about Program Invariants? [2023-ICML]

#### proof generation

Selene: Pioneering Automated Proof in Software Verification [2024-ACL]

#### Resource leak detection

Boosting Static Resource Leak Detection via LLM-based Resource-Oriented Intention Inference [2023-arXiv]

#### taint analysis

Harnessing the Power of LLM to Support Binary Taint Analysis [2023-arXiv]

#### Actionable Warning Identification

Pre-trained Model-based Actionable Warning Identification: A Feasibility Study [2024-arXiv]

#### Adversarial Attack

An LLM-Assisted Easy-to-Trigger Backdoor Attack on Code Completion Models: Injecting Disguised Vulnerabilities against Strong Detection [2024-USENIX Security]

CodeBERT‐Attack: Adversarial attack against source code deep learning models via pre‐trained model [2023-JSEP]

ChatGPT as an Attack Tool: Stealthy Textual Backdoor Attack via Blackbox Generative Model Trigger [2024-NAACL]

#### API Misuse Detection

1. Exploring Automatic Cryptographic API Misuse Detection in the Era of LLMs [2024-arXiv]


#### API Testing

1. KAT: Dependency-aware Automated API Testing with Large Language Models [2024-ICST]


#### Assertion Generation

1. TOGA: A Neural Method for Test Oracle Generation [2022-ICSE]  
2. Can Large Language Models Transform Natural Language Intent into Formal Method Postconditions? [2024-FSE]  
3. Beyond Code Generation: Assessing Code LLM Maturity with Postconditions [2024-arXiv]  
4. An Empirical Study on Focal Methods in Deep-Learning-Based Approaches for Assertion Generation [2024-ICSE]  
5. TOGLL: Correct and Strong Test Oracle Generation with LLMs [2024-arXiv]  
6. ChIRAAG: ChatGPT Informed Rapid and Automated Assertion Generation [2024-arXiv]  
7. Retrieval-Based Prompt Selection for Code-Related Few-Shot Learning [2023-ICSE]  
8. AssertionBench: A Benchmark to Evaluate Large-Language Models for Assertion Generation [2024-NeurIPS]  
9. Generating Accurate Assert Statements for Unit Test Cases using Pretrained Transformers [2022-AST]  
10. Chat-like Asserts Prediction with the Support of Large Language Model [2024-arXiv]  

#### Binary Code Similarity Detection

1. Practical Binary Code Similarity Detection with BERT-based Transferable Similarity Learning [2022-ACSAC]  
2. CRABS-former: CRoss-Architecture Binary Code Similarity Detection based on Transformer [2024-Internetware]  
3. jTrans: Jump-Aware Transformer for Binary Code Similarity Detection [2022-ISSTA]  
4. Order Matters: Semantic-Aware Neural Networks for Binary Code Similarity Detection [2020-AAAI]  

#### Code Execution

1. SelfPiCo: Self-Guided Partial Code Execution with LLMs [2024-ISSTA]


#### Decompilation

1. SLaDe: A Portable Small Language Model Decompiler for Optimized Assembly [2024-CGO]  
2. DeGPT: Optimizing Decompiler Output with LLM [2024-NDSS]  
3. Nova+: Generative Language Models for Binaries [2023-arXiv]  
4. How Far Have We Gone in Binary Code Understanding Using Large Language Models [2024-ICSME]  
5. WaDec: Decompile WebAssembly Using Large Language Model [2024-arXiv]  
6. Refining Decompiled C Code with Large Language Models [2023-arXiv]  
7. LmPa: Improving Decompilation by Synergy of Large Language Model and Program Analysis [2023-arXiv]  
8. LLM4Decompile: Decompiling Binary Code with Large Language Models [2024-arXiv]  

#### Failure-Inducing Testing

1. Nuances are the Key: Unlocking ChatGPT to Find Failure-Inducing Tests with Differential Prompting [2023-ASE]


#### Fault Localization

1. LLM Fault Localisation within Evolutionary Computation Based Automated Program Repair [2024-GECCO]  
2. Supporting Cross-language Cross-project Bug Localization Using Pre-trained Language Models [2024-arXiv]  
3. Too Few Bug Reports? Exploring Data Augmentation for Improved Changeset-based Bug Localization [2023-arXiv]  
4. Fast Changeset-based Bug Localization with BERT [2022-ICSE]  
5. Pre-training Code Representation with Semantic Flow Graph for Effective Bug Localization [2023-FSE]  
6. Impact of Large Language Models of Code on Fault Localization [2024-arXiv]  
7. A Quantitative and Qualitative Evaluation of LLM-based Explainable Fault Localization [2024-FSE]  
8. Enhancing Bug Localization Using Phase-Based Approach [2023-IEEE Access]  
9. AgentFL: Scaling LLM-based Fault Localization to Project-Level Context [2024-arXiv]  
10. Face It Yourselves: An LLM-Based Two-Stage Strategy to Localize Configuration Errors via Logs [2024-ISSTA]  
11. Demystifying Faulty Code with LLM: Step-by-Step Reasoning for Explainable Fault Localization [2024-arXiv]  
12. Demystifying faulty code: Step-by-step reasoning for explainable fault localization [2024-SANER]  
13. Large Language Models in Fault Localisation [2023-arXiv]  
14. Better Debugging: Combining Static Analysis and LLMs for Explainable Crashing Fault Localization [2024-arXiv]  
15. Large language models for test-free fault localization [2024-ICSE]  
16. TroBo: A Novel Deep Transfer Model for Enhancing Cross-Project Bug Localization [2021-KSEM]  
17. ConDefects: A Complementary Dataset to Address the Data Leakage Concern for LLM-Based Fault Localization and Program Repair [2024-FSE]  

#### Fuzzing

1. SearchGEM5: Towards Reliable gem5 with Search Based Software Testing and Large Language Models [2023-SSBSE]  
2. Large Language Models are Zero-Shot Fuzzers: Fuzzing Deep-Learning Libraries via Large Language Models [2023-ISSTA]  
3. Large Language Models are Edge-Case Fuzzers: Testing Deep Learning Libraries via FuzzGPT [2023-ICSE]  
4. Large Language Models are Edge-Case Generators- Crafting Unusual Programs for Fuzzing Deep Learning Libraries [2023-ICSE]  
5. CovRL: Fuzzing JavaScript Engines with Coverage-Guided Reinforcement Learning for LLM-based Mutation [2024-arXiv]  
6. Fuzzing JavaScript Interpreters with Coverage-Guided Reinforcement Learning for LLM-based Mutation [2024-ISSTA]  
7. Augmenting Greybox Fuzzing with Generative AI [2023-arXiv]  
8. Large Language Model guided Protocol Fuzzing [2023-NDSS]  
9. Fuzzing BusyBox: Leveraging LLM and Crash Reuse for Embedded Bug Unearthing [2024-USENIX Security]  
10. Large Language Models for Fuzzing Parsers [2023-Fuzzzing]  
11. Llm4fuzz: Guided fuzzing of smart contracts with large language models [2024-arXiv]  
12. Llmif: Augmented large language model for fuzzing iot devices [2024-SP]  
13. Fuzz4all: Universal fuzzing with large language models [2024-ICSE]  
14. Kernelgpt: Enhanced kernel fuzzing via large language models [2023-arXiv]  
15. White-box Compiler Fuzzing Empowered by Large Language Models [2023-arXiv]  
16. LLAMAFUZZ: Large Language Model Enhanced Greybox Fuzzing [2024-arXiv]  
17. Understanding Large Language Model Based Fuzz Driver Generation [2023-arXiv]  
18. How Effective Are They? Exploring Large Language Model Based Fuzz Driver Generation [2024-ISSTA]  

#### GUI Testing

1. Vision-driven Automated Mobile GUI Testing via Multimodal Large Language Model [2024-arXiv]  
2. Fill in the Blank: Context-aware Automated Text Input Generation for Mobile GUI Testing [2022-ICSE]  
3. Make LLM a Testing Expert: Bringing Human-like Interaction to Mobile GUI Testing via Functionality-aware Decisions [2023-ICSE]  
4. Autonomous Large Language Model Agents Enabling Intent-Driven Mobile GUI Testing [2023-arXiv]  
5. Intent-Driven Mobile GUI Testing with Autonomous Large Language Model Agents [2024-ICST]  
6. Guardian: A Runtime Framework for LLM-based UI Exploration [2024-ISSTA]  

#### Indirect Call Analysis

1. Semantic-Enhanced Indirect Call Analysis with Large Language Models [2024-arXiv]

#### Mutation Testing

1. µBert- Mutation Testing using Pre-Trained Language Models [2022-ICST]  
2. On the Coupling between Vulnerabilities and LLM-generated Mutants: A Study on Vul4J dataset [2024-ICST]  
3. Llm-guided formal verification coupled with mutation testing [2024-DATE]  
4. Automated Bug Generation in the era of Large Language Models [2023-arXiv]  
5. Contextual Predictive Mutation Testing [2023-FSE]  
6. Efficient Mutation Testing via Pre-Trained Language Models [2023-arXiv]  
7. Mutation-based consistency testing for evaluating the code understanding capability of llms [2024-FSE]  
8. VULGEN: Realistic Vulnerability Generation Via Pattern Mining and Deep Learning [2023-ICSE]  
9. Learning Realistic Mutations- Bug Creation for Neural Bug Detectors [2022-ICST]  
10. Large Language Models for Equivalent Mutant Detection: How Far are We? [2024-ISSTA]  
11. LLMorpheus: Mutation Testing using Large Language Models [2024-arXiv]  
12. An Exploratory Study on Using Large Language Models for Mutation Testing [2024-arXiv]  

#### NLP Testing

1. Machine Translation Testing via Pathological Invariance [2020-FSE]  
2. Structure-Invariant Testing for Machine Translation [2020-ICSE]  
3. Dialtest: automated testing for recurrent-neural-network-driven dialogue systems [2021-ISSTA]  
4. Qatest: A uniform fuzzing framework for question answering systems [2022-ASE]  
5. Improving Machine Translation Systems via Isotopic Replacement [2022-ICSE]  
6. Mttm: Metamorphic testing for textual content moderation software [2023-ICSE]  
7. Automated testing and improvement of named entity recognition systems [2023-FSE]  

#### Penetration Testing

1. PentestGPT: An LLM-empowered Automatic Penetration Testing Tool [2024-USENIX Security]
2. Getting pwn'd by AI: Penetration Testing with Large Language Models [2023-FSE]
3. CIPHER: Cybersecurity Intelligent Penetration-testing Helper for Ethical Researcher [2024-arXiv]
4. PTGroup: An Automated Penetration Testing Framework Using LLMs and Multiple Prompt Chains [2024-ICIC]


#### Program Analysis

1. CFStra: Enhancing Configurable Program Analysis Through LLM-Driven Strategy Selection Based on Code Features [2024-TASE]

#### Program Reduction

1. LPR: Large Language Models-Aided Program Reduction [2024-ISSTA]

#### Property-based Testing 

1. Can Large Language Models Write Good Property-Based Tests? [2023-arXiv]

#### Simulation Testing

1. DiaVio: LLM-Empowered Diagnosis of Safety Violations in ADS Simulation Testing [2024-ISSTA]

#### Static Analysis

1. Interleaving Static Analysis and LLM Prompting [2024-SOAP]  
2. Large Language Models for Code Analysis: Do LLMs Really Do Their Job? [2024-USENIX Security]  
3. E&V: Prompting Large Language Models to Perform Static Analysis by Pseudo-code Execution and Verification [2023-arXiv]  
4. Assisting Static Analysis with Large Language Models: A ChatGPT Experiment [2023-FSE]  
5. Enhancing Static Analysis for Practical Bug Detection: An LLM-Integrated Approach [2024-OOPSLA]  
6. SkipAnalyzer: An Embodied Agent for Code Analysis with Large Language Models [2023-arXiv]  

#### Static Warning Validating

1. Automatically Inspecting Thousands of Static Bug Warnings with Large Language Model: How Far Are We? [2024-TOSEM]
2. Analyzing source code vulnerabilities in the D2A dataset with ML ensembles and C-BERT [2024-EMSE]

#### Test Generation

1. Generating Test Scenarios from NL Requirements using Retrieval-Augmented LLMs: An Industrial Study [2024-arXiv]  
2. ChatGPT is a Remarkable Tool—For Experts [2023-Data Intelligence]  
3. Large Language Models for Mobile GUI Text Input Generation: An Empirical Study [2024-arXiv]  
4. Effective Test Generation Using Pre-trained Large Language Models and Mutation Testing [2023-IST]  
5. Leveraging Large Language Models for Enhancing the Understandability of Generated Unit Tests [2024-ICSE]  
6. Mokav: Execution-driven Differential Testing with LLMs [2024-arXiv]  
7. TestART: Improving LLM-based Unit Test via Co-evolution of Automated Generation and Repair Iteration [2024-arXiv]  
8. An initial investigation of ChatGPT unit test generation capability [2023-SAST]  
9. Exploring Fuzzing as Data Augmentation for Neural Test Generation [2024-arXiv]  
10. Harnessing the Power of LLMs: Automating Unit Test Generation for High-Performance Computing [2024-arXiv]  
11. Navigating Confidentiality in Test Automation: A Case Study in LLM Driven Test Data Generation [2024-SANER]  
12. ChatGPT and Human Synergy in Black-Box Testing: A Comparative Analysis [2024-arXiv]  
13. CODAMOSA: Escaping Coverage Plateaus in Test Generation with Pre-trained Large Language Models [2023-ICSE]  
14. DLLens: Testing Deep Learning Libraries via LLM-aided Synthesis [2024-arXiv]  
15. Large Language Models as Test Case Generators: Performance Evaluation and Enhancement [2024-arXiv]  
16. Leveraging Large Language Models for Automated Web-Form-Test Generation: An Empirical Study [2024-arXiv]  
17. LLM-Powered Test Case Generation for Detecting Tricky Bugs [2024-arXiv]  
18. A System for Automated Unit Test Generation Using Large Language Models and Assessment of Generated Test Suites [2024-arXiv]  
19. Code Agents are State of the Art Software Testers [2024-arXiv]  
20. Test Code Generation for Telecom Software Systems using Two-Stage Generative Model [2024-arXiv]  
21. CasModaTest: A Cascaded and Model-agnostic Self-directed Framework for Unit Test Generation [2024-arXiv]  
22. Large-scale, Independent and Comprehensive study of the power of LLMs for test case generation [2024-arXiv]  
23. CoverUp: Coverage-Guided LLM-Based Test Generation [2024-arXiv]  
24. Automatic Generation of Test Cases based on Bug Reports: a Feasibility Study with Large Language Models [2024-ICSE]  
25. CAT-LM Training Language Models on Aligned Code And Tests [2023-ASE]  
26. Code-aware prompting: A study of coverage guided test generation in regression setting using llm [2024-FSE]  
27. Adaptive test generation using a large language model [2023-arXiv]  
28. An Empirical Evaluation of Using Large Language Models for Automated Unit Test Generation [2024-TSE]  
29. Domain Adaptation for Deep Unit Test Case Generation [2023-arXiv]  
30. Exploring the effectiveness of large language models in generating unit tests [2023-arXiv]  
31. Reinforcement Learning from Automatic Feedback for High-Quality Unit Test Generation [2023-arXiv]  
32. ChatGPT vs SBST: A Comparative Assessment of Unit Test Suite Generation [2024-TSE]  
33. Unit Test Case Generation with Transformers and Focal Context [2020-arXiv]  
34. HITS: High-coverage LLM-based Unit Test Generation via Method Slicing [2024-ASE]  
35. Optimizing Search-Based Unit Test Generation with Large Language Models: An Empirical Study [2024-Internetware]  
36. ChatUniTest: a ChatGPT-based automated unit test generation tool [2023-arXiv]  
37. The Program Testing Ability of Large Language Models for Code [2023-arXiv]  
38. An Empirical Study of Unit Test Generation with Large Language Models [2024-ASE]  
39. Enhancing LLM-based Test Generation for Hard-to-Cover Branches via Program Analysis [2024-arXiv]  
40. No More Manual Tests? Evaluating and Improving ChatGPT for Unit Test Generation [2023-arXiv]  
41. Evaluating and Improving ChatGPT for Unit Test Generation [2024-FSE]  
42. Algo: Synthesizing algorithmic programs with generated oracle verifiers [2023-NeurIPS]  
43. How well does LLM generate security tests? [2023-arXiv]  
44. An LLM-based Readability Measurement for Unit Tests' Context-aware Inputs [2024-arXiv]  
45. A3Test - Assertion Augmented Automated Test Case Generation [2024-IST]  
46. LLM4Fin: Fully Automating LLM-Powered Test Case Generation for FinTech Software Acceptance Testi [2024-ISSTA]  

#### Test Suite Minimization

1. LTM: Scalable and Black-box Similarity-based Test Suite Minimization based on Language Models [2023-arXiv]

#### Vulnerability Detection

1. FLAG: Finding Line Anomalies (in code) with Generative AI [2023-arXiv]  
2. Vulnerability Detection and Monitoring Using LLM [2023-WIECON-ECE]  
3. Low Level Source Code Vulnerability Detection Using Advanced BERT Language Model [2022-Canadian AI]  
4. LLM-based Vulnerability Sourcing from Unstructured Data [2024-EuroS&PW]  
5. Transformer-based vulnerability detection in code at EditTime: Zero-shot, few-shot, or fine-tuning? [2023-arXiv]  
6. Diversevul: A new vulnerable source code dataset for deep learning based vulnerability detection [2023-RAID]  
7. Bridge and Hint: Extending Pre-trained Language Models for Long-Range Code [2024-arXiv]  
8. LLM-Enhanced Static Analysis for Precise Identification of Vulnerable OSS Versions [2024-arXiv]  
9. VulCatch: Enhancing Binary Vulnerability Detection through CodeT5 Decompilation and KAN Advanced Feature Extraction [2024-arXiv]  
10. Exploring RAG-based Vulnerability Augmentation with LLMs [2024-arXiv]  
11. Vulnerability Detection with Code Language Models: How Far Are We? [2024-arXiv]  
12. Optimizing software vulnerability detection using RoBERTa and machine learning [2024-AUSE]  
13. Large Language Models for Secure Code Assessment: A Multi-Language Empirical Study [2024-arXiv]  
14. Generalization-Enhanced Code Vulnerability Detection via Multi-Task Instruction Fine-Tuning [2024-ACL]  
15. Vul-RAG: Enhancing LLM-based Vulnerability Detection via Knowledge-level RAG [2024-arXiv]  
16. LineVul: A Transformer-based Line-Level Vulnerability Prediction [2021-MSR]  
17. How Far Have We Gone in Vulnerability Detection Using Large Language Models [2023-arXiv]  
18. BERT-and TF-IDF-based feature extraction for long-lived bug prediction in FLOSS: a comparative study [2023-IST]  
19. SCoPE: Evaluating LLMs for Software Vulnerability Detection [2024-arXiv]  
20. The EarlyBIRD Catches the Bug: On Exploiting Early Layers of Encoder Models for More Efficient Code Classification [2023-FSE]  
21. Vulberta: Simplified source code pre-training for vulnerability detection [2022-IJCNN]  
22. DFEPT: Data Flow Embedding for Enhancing Pre-Trained Model Based Vulnerability Detection [2024-Internetware]  
23. A Study of Using Multimodal LLMs for Non-Crash Functional Bug Detection in Android Apps [2024-arXiv]  
24. Understanding the Effectiveness of Large Language Models in Detecting Security Vulnerabilities [2023-arXiv]  
25. A Qualitative Study on Using ChatGPT for Software Security: Perception vs. Practicality [2024-arXiv]  
26. Detecting Phishing Sites Using ChatGPT [2023-arXiv]  
27. Bug In the Code Stack: Can LLMs Find Bugs in Large Python Code Stacks [2024-arXiv]  
28. LLM-Assisted Static Analysis for Detecting Security Vulnerabilities [2024-arXiv]  
29. VulDetectBench: Evaluating the Deep Capability of Vulnerability Detection with Large Language Models [2024-arXiv]  
30. EaTVul: ChatGPT-based Evasion Attack Against Software Vulnerability Detection [2024-arXiv]  
31. GRACE: Empowering LLM-based software vulnerability detection with graph structure and in-context learning [2024-JSS]  
32. Evaluating Large Language Models in Detecting Test Smells [2024-arXiv]  
33. Harnessing the Power of LLMs in Source Code Vulnerability Detection [2024-arXiv]  
34. Multi-role Consensus through LLMs Discussions for Vulnerability Detection [2024-arXiv]  
35. Towards Effectively Detecting and Explaining Vulnerabilities Using Large Language Models [2024-arXiv]  
36. Llbezpeky: Leveraging large language models for vulnerability detection [2024-arXiv]  
37. Can Large Language Models Find And Fix Vulnerable Software? [2023-arXiv]  
38. Chain-of-Thought Prompting of Large Language Models for Discovering and Fixing Software Vulnerabilities [2024-arXiv]  
39. Automated Software Vulnerability Static Code Analysis Using Generative Pre-Trained Transformer Models [2024-arXiv]  
40. XGV-BERT: Leveraging Contextualized Language Model and Graph Neural Network for Efficient Software Vulnerability Detection [2023-arXiv]  
41. ALPINE: An adaptive language-agnostic pruning method for language models for code [2024-arXiv]  
42. Finetuning Large Language Models for Vulnerability Detection [2024-arXiv]  
43. A Comprehensive Study of the Capabilities of Large Language Models for Vulnerability Detection [2024-arXiv]  
44. Dataflow Analysis-Inspired Deep Learning for Efficient Vulnerability Detection [2024-ICSE]  
45. An Empirical Study of Deep Learning Models for Vulnerability Detection [2023-ICSE]  
46. DexBERT: Effective, Task-Agnostic and Fine-grained Representation Learning of Android Bytecode [2023-TSE]  
47. GPTScan: Detecting Logic Vulnerabilities in Smart Contracts by Combining GPT with Program Analysis [2024-ICSE]  
48. LLM4Vuln: A Unified Evaluation Framework for Decoupling and Enhancing LLMs' Vulnerability Reasoning [2024-arXiv]  
49. Using large language models to better detect and handle software vulnerabilities and cyber security threats [2024-arXiv]  
50. Harnessing Large Language Models for Software Vulnerability Detection: A Comprehensive Benchmarking Study [2024-arXiv]  
51. CSGVD: a deep learning approach combining sequence and graph embedding for source code vulnerability detection [2023-JSS]  
52. Just-in-Time Security Patch Detection -- LLM At the Rescue for Data Augmentation [2023-arXiv]  
53. Transformer-based language models for software vulnerability detection [2022-ACSAC]  
54. Can Large Language Models Identify And Reason About Security Vulnerabilities? Not Yet [2023-arXiv]  
55. Bridging the Gap: A Study of AI-based Vulnerability Management between Industry and Academia [2024-arXiv]  
56. Code Structure-Aware through Line-level Semantic Learning for Code Vulnerability Detection [2024-arXiv]  
57. M2CVD: Multi-Model Collaboration for Code Vulnerability Detection [2024-arXiv]  
58. VulEval: Towards Repository-Level Evaluation of Software Vulnerability Detection [2024-arXiv]  
59. Natural Language Generation and Understanding of Big Code for AI-Assisted Programming: A Review [2023-Entropy]  
60. Peculiar: Smart Contract Vulnerability Detection Based on Crucial Data Flow Graph and Pre-training Techniques [2022-ISSRE]  
61. DLAP: A Deep Learning Augmented Large Language Model Prompting Framework for Software Vulnerability Detection [2024-arXiv]  
62. Security Vulnerability Detection with Multitask Self-Instructed Fine-Tuning of Large Language Models [2024-arXiv]  
63. Multitask-based Evaluation of Open-Source LLM on Software Vulnerability [2024-arXiv]  
64. Pros and Cons! Evaluating ChatGPT on Software Vulnerability [2024-arXiv]  
65. Security Code Review by LLMs: A Deep Dive into Responses [2024-arXiv]  
66. Enhancing Deep Learning-based Vulnerability Detection by Building Behavior Graph Model [2023-ICSE]  
67. Prompt-Enhanced Software Vulnerability Detection Using ChatGPT [2023-arXiv]  
68. Coding-PTMs: How to Find Optimal Code Pre-trained Models for Code Embedding in Vulnerability Detection? [2024-arXiv]  
69. Comparison of Static Application Security Testing Tools and Large Language Models for Repo-level Vulnerability Detection [2024-arXiv]  
70. Large Language Model for Vulnerability Detection and Repair: Literature Review and Roadmap [2024-arXiv]  
71. Large language model for vulnerability detection: Emerging results and future directions [2024-ICSE]  
72. An exploratory study on just-in-time multi-programming-language bug prediction [2024-IST]  
73. Assessing the Effectiveness of Vulnerability Detection via Prompt Tuning: An Empirical Study [2023-APSEC]  
74. Detecting Common Weakness Enumeration Through Training the Core Building Blocks of Similar Languages Based on the CodeBERT Model [2023-APSEC]  
75. Improving long-tail vulnerability detection through data augmentation based on large language models [2024-ICSME]  
76. Large Language Models can Connect the Dots: Exploring Model Optimization Bugs with Domain Knowledge-aware Prompts [2024-ISSTA]  

#### Vulnerable Dependency Alert Detection

1. Silent Vulnerable Dependency Alert Prediction with Vulnerability Key Aspect Explanation [2023-ICSE]

#### Theorem Proving

1. LLM-Enhanced Theorem Proving with Term Explanation and Tactic Parameter Repair [2024-Internetware]

### 📱Software Maintenance

#### Android permissions

1. Large Language Model vs. Stack Overflow in Addressing Android Permission Related Challenges [2024-MSR]

#### APP Review Analysis

1. T-FREX: A Transformer-based Feature Extraction Method from Mobile App Reviews [2024-SANER]
2. Where is Your App Frustrating Users? [2022-ICSE]

#### Bug Report Detection

1. Duplicate bug report detection by using sentence embedding and fine-tuning [2021-ICSME]
2. Can LLMs Demystify Bug Reports? [2023-arXiv]
3. Refining GPT-3 Embeddings with a Siamese Structure for Technical Post Duplicate Detection [2024-SANER]
4. Cupid: Leveraging ChatGPT for More Accurate Duplicate Bug Report Detection [2023-arXiv]
5. Few-shot learning for sentence pair classification and its applications in software engineering [2023-arXiv]

#### Bug Reproduction

1. Prompting Is All Your Need: Automated Android Bug Replay with Large Language Models [2023-ICSE]
2. CrashTranslator: Automatically Reproducing Mobile Application Crashes Directly from Stack Trace [2024-ICSE]
3. Evaluating Diverse Large Language Models for Automatic and General Bug Reproduction [2023-arXiv]
4. Large Language Models are Few-shot Testers: Exploring LLM-based General Bug Reproduction [2023-ICSE]

#### Bug Triaging

1. Neighborhood contrastive learning-based graph neural network for bug triaging [2024-SCP]
2. A Comparative Study of Transformer-based Neural Text Representation Techniques on Bug Triaging [2023-ASE]
3. A Light Bug Triage Framework for Applying Large Pre-trained Language Model [2022-ASE]

#### Code Clone Detection

1. GPTCloneBench: A comprehensive benchmark of semantic clones and cross-language clones using GPT-3 model and SemanticCloneBench [2023-ICSME]
2. Using a Nearest-Neighbour, BERT-Based Approach for Scalable Clone Detection [2022-ICSME]
3. Towards Understanding the Capability of Large Language Models on Code Clone Detection: A Survey [2023-arXiv]
4. AdaCCD: Adaptive Semantic Contrasts Discovery Based Cross Lingual Adaptation for Code Clone Detection [2024-AAAI]
5. Investigating the Efficacy of Large Language Models for Code Clone Detection [2024-ICPC]
6. Large Language Models for cross-language code clone detection [2024-arXiv]
7. Utilization of Pre-trained Language Model for Adapter-based Knowledge Transfer in Software Engineering [2023-EMSE]
8. An exploratory study on code attention in bert [2022-ICPC]
9. Assessing the Code Clone Detection Capability of Large Language Models [2024-ICCQ]
10. Interpreting CodeBERT for Semantic Code Clone Detection [2023-APSEC]

#### Code Coverage Prediction

1. Predicting Code Coverage without Execution [2023-arXiv]

#### Code Evolution

1. Multilingual Code Co-Evolution Using Large Language Models [2023-FSE]

#### Code Porting

1. Enabling Memory Safety of C Programs using LLMs [2024-arXiv]
2. Hybrid API Migration: A Marriage of Small API Mapping Models and Large Language Models [2023-Internetware]

#### Code Refactoring

1. RefBERT: A Two-Stage Pre-trained Framework for Automatic Rename Refactoring [2023-ISSTA]
2. Next-Generation Refactoring: Combining LLM Insights and IDE Capabilities for Extract Method [2024-ICSME]
3. Refactoring Programs Using Large Language Models with Few-Shot Examples [2023-APSEC]
4. Refactoring to Pythonic Idioms: A Hybrid Knowledge-Driven Approach Leveraging Large Language Models [2024-FSE]

#### Code Review

1. Can LLMs Replace Manual Annotation of Software Engineering Artifacts? [2024-arXiv]
2. Improving the learning of code review successive tasks with cross-task knowledge distillation [2024-FSE]
3. A GPT-based Code Review System for Programming Language Learning [2024-arXiv]
4. Exploring the Capabilities of LLMs for Code Change Related Tasks [2024-arXiv]
5. Incivility Detection in Open Source Code Review and Issue Discussions [2024-JSS]
6. Augmenting commit classification by using fine-grained source code changes and a pre-trained deep neural language model [2021-IST]
7. Exploring the Potential of ChatGPT in Automated Code Refinement: An Empirical Study [2023-ICSE]
8. Automated Summarization of Stack Overflow Posts [2023-ICSE]
9. Evaluating Language Models for Generating and Judging Programming Feedback [2024-arXiv]
10. AUGER: automatically generating review comments with pre-training models [2022-FSE]
11. Automating code review activities by large-scale pre-training [2022-FSE]
12. Improving Code Refinement for Code Review Via Input Reconstruction and Ensemble Learning [2023-APSEC]
13. LLaMA-Reviewer: Advancing code review automation with large language models through parameter-efficient fine-tuning [2023-ISSRE]
14. LLM Critics Help Catch LLM Bugs [2024-arXiv]
15. Fine-tuning and prompt engineering for large language models-based code review automation [2024-IST]
16. AI-powered Code Review with LLMs: Early Results [2024-arXiv]
17. A Multi-Step Learning Approach to Assist Code Review [2023-SANER]
18. Code Review Automation: Strengths and Weaknesses of the State of the Art [2024-TSE]
19. Using Pre-Trained Models to Boost Code Review Automation [2022-ICSE]
20. AI-Assisted Assessment of Coding Practices in Modern Code Review [2024-arXiv]
21. Explaining Explanation: An Empirical Study on Explanation in Code Reviews [2023-arXiv]
22. Aspect-based api review classification: How far can pre-trained transformer model go? [2022-SANER]
23. Automatic Code Review by Learning the Structure Information of Code Graph [2023-Sensors]
24. The Right Prompts for the Job: Repair Code-Review Defects with Large Language Model [2023-arXiv]

#### Code Smells

1. Pre-trained Model Based Feature Envy Detection [2023-MSR]

#### Commit Message Generation

1. Commitbert: Commit message generation using pre-trained programming language mode [2021-nlp4prog]
2. Only diff is Not Enough: Generating Commit Messages Leveraging Reasoning and Action of Large Language Model [2024-FSE]
3. Commit Messages in the Age of Large Language Models [2024-arXiv]
4. Automated Commit Message Generation with Large Language Models: An Empirical Study and Beyond [2024-arXiv]
5. Automatic Commit Message Generation: A Critical Review and Directions for Future Work [2024-TSE]

#### Compiler Optimization

1. Large Language Models for Compiler Optimization [2023-arXiv]
2. Meta Large Language Model Compiler: Foundation Models of Compiler Optimization [2024-arXiv]
3. ViC: Virtual Compiler Is All You Need For Assembly Code Search [2024-arXiv]
4. Priority Sampling of Large Language Models for Compilers [2024-arXiv]
5. Should AI Optimize Your Code? A Comparative Study of Current Large Language Models Versus Classical Optimizing Compilers [2024-arXiv]
6. Learning Performance-Improving Code Edits [2023-ICLR]
7. Isolating Compiler Bugs by Generating Effective Witness Programs With Large Language Models [2024-TSE]
8. Iterative or Innovative? A Problem-Oriented Perspective for Code Optimization [2024-arXiv]

#### Debugging

1. Explainable Automated Debugging via Large Language Model-driven Scientific Debugging [2023-arXiv]

#### Exception Handling Recommendation

Programming Assistant for Exception Handling with CodeBERT [2024-ICSE]

#### Flaky Test Prediction

Flakify: a black-box, language model-based predictor for flaky tests [2022-TSE]

#### Incident Management

1. Recommending Root-Cause and Mitigation Steps for Cloud Incidents using Large Language Models [2023-ICSE]
2. Xpert: Empowering Incident Management with Query Recommendations via Large Language Models [2024-ICSE]

#### Issue Labeling

1. Impact of data quality for automatic issue classification using pre-trained language models [2024-JSS]
2. Leveraging GPT-like LLMs to Automate Issue Labeling [2024-MSR]

#### Log Analysis

1. GLOSS: Guiding Large Language Models to Answer Questions from System Logs [2024-SANER]
2. ULog: Unsupervised Log Parsing with Large Language Models through Log Contrastive Units [2024-arXiv]
3. LILAC: Log Parsing using LLMs with Adaptive Parsing Cache [2024-FSE]
4. Log Parsing with Prompt-based Few-shot Learning [2023-ICSE]
5. Log Parsing: How Far Can ChatGPT Go? [2023-ASE]
6. Exploring the Effectiveness of LLMs in Automated Logging Generation: An Empirical Study [2023-arXiv]
7. Interpretable Online Log Analysis Using Large Language Models with Prompt Strategies [2024-ICPC]
8. LogPrompt: Prompt Engineering Towards Zero-Shot and Interpretable Log Analysis [2023-arXiv]
9. KnowLog: Knowledge Enhanced Pre-trained Language Model for Log Understanding [2024-ICSE]
10. LLMParser: An Exploratory Study on Using Large Language Models for Log Parsing [2024-ICSE]
11. Using deep learning to generate complete log statements [2022-ICSE]
12. Log statements generation via deep learning: Widening the support provided to developers [2024-JSS]
13. The Effectiveness of Compact Fine-Tuned LLMs for Log Parsing [2024-ICSME]
14. An Assessment of ChatGPT on Log Data [2023-AIGC]
15. LogStamp: Automatic Online Log Parsing Based on Sequence Labelling [2022-SIGMETRICS]
16. Log Parsing with Self-Generated In-Context Learning and Self-Correction [2024-arXiv]
17. Stronger, Faster, and Cheaper Log Parsing with LLMs [2024-arXiv]
18. UniLog: Automatic Logging via LLM and In-Context Learning [2024-ICSE]
19. Log Parsing with Generalization Ability under New Log Types [2023-FSE]
20. A Comparative Study on Large Language Models for Log Parsing [2024-ESEM]


#### Log Anomaly Detection

1. Log Sequence Anomaly Detection based on Template and Parameter Parsing via BERT [2024-TDSC]
2. LogBERT: Log Anomaly Detection via BERT [2021-IJCNN]
3. Anomaly Detection on Unstable Logs with GPT Models [2024-arXiv]
4. Parameter-Efficient Log Anomaly Detection based on Pre-training model and LORA [2023-ISSRE]
5. Hitanomaly: Hierarchical transformers for anomaly detection in system log [2020-TNSM]
6. LAnoBERT: : System log anomaly detection based on BERT masked language model [2023-Applied Soft Computing]
7. Swisslog: Robust anomaly detection and localization for interleaved unstructured logs [2023-TDSC]
8. LogBD: A Log Anomaly Detection Method Based on Pretrained Models and Domain Adaptation [2023-Applied Sciences]
9. On the Influence of Data Resampling for Deep Learning-Based Log Anomaly Detection: Insights and Recommendations [2024-arXiv]
10. LogEncoder: Log-Based Contrastive Representation Learning for Anomaly Detection [2023-TNSM]
11. LogGPT: Exploring ChatGPT for Log-Based Anomaly Detection [2023-HPCC/DSS/SmartCity/DependSys]
12. DeepUserLog Deep Anomaly Detection on User Log Using Semantic Analysis and Key Value Data [2023-ISSRE]
13. Allinfolog: Robust diverse anomalies detection based on all log features [2022-TNSM]

#### Malware Tracker

1. Maltracker: A Fine-Grained NPM Malware Tracker Copiloted by LLM-Enhanced Dataset [2024-ISSTA]

#### Mobile app crash detection

1. Testing the Limits: Unusual Text Inputs Generation for Mobile App Crash Detection with Large Language Model [2023-ICSE]

#### Outage Understanding

1. Assess and Summarize: Improve Outage Understanding with Large Language Models [2023-FSE]

#### Patch Correctness Assessment

1. Improving Patch Correctness Analysis via Random Testing and Large Language Models [2024-ICST]
2. Evaluating representation learning of code changes for predicting patch correctness in program repair [2020-ASE]
3. Is this Change the Answer to that Problem? Correlating Descriptions of Bug and Code Changes for Evaluating Patch Correctness [2022-ASE]
4. The Best of Both Worlds: Combining Learned Embeddings with Engineered Features for Accurate Prediction of Correct Patches [2023-TOSEM]
5. APPT Boosting Automated Patch Correctness Prediction via Pre-trained Language Model [2023-TSE]
6. PatchZero: Zero-Shot Automatic Patch Correctness Assessment [2023-arXiv]

#### Privacy Policy

1. A Large Language Model Approach to Code and Privacy Policy Alignment [2024-SANER]

#### Program Repair

1. TFix: Learning to Fix Coding Errors with a Text-to-Text Transformer [2021-ICML]
2. RepairAgent: An Autonomous, LLM-Based Agent for Program Repair [2024-arXiv]
3. A study on Prompt Design, Advantages and Limitations of ChatGPT for Deep Learning Program Repair [2023-arXiv]
4. Automated Repair of AI Code with Large Language Models and Formal Verification [2024-arXiv]
5. PyTy: Repairing Static Type Errors in Python [2024-ICSE]
6. MergeRepair: An Exploratory Study on Merging Task-Specific Adapters in Code LLMs for Automated Program Repair [2024-arXiv]
7. Fixing Rust Compilation Errors using LLMs [2023-arXiv]
8. Generating Bug-Fixes Using Pretrained Transformers [2021-PLDI]
9. Resolving Crash Bugs via Large Language Models: An Empirical Study [2022-arXiv]
10. Automated Repair of Programs from Large Language Models [2022-ICSE]
11. Baldur: Whole-Proof Generation and Repair with Large Language Models [2023-FSE]
12. T5APR: Empowering automated program repair across languages through checkpoint ensemble [2024-JSS]
13. Shipwright: A Human-in-the-Loop System for Dockerfile Repair [2021-ICSE]
14. CigaR: Cost-efficient Program Repair with LLMs [2024-arXiv]
15. A Deep Dive into Large Language Models for Automated Bug Localization and Repair [2024-FSE]
16. A Chain of AI-based Solutions for Resolving FQNs and Fixing Syntax Errors in Partial Code [2023-arXiv]
17. An empirical study on fine-tuning large language models of code for automated program repair [2023-ASE]
18. Code Security Vulnerability Repair Using Reinforcement Learning with Large Language Models [2024-arXiv]
19. CURE Code-Aware Neural Machine Translation for Automatic Program Repair [2021-ICSE]
20. Impact of Code Language Models on Automated Program Repair [2023-ICSE]
21. InferFix: End-to-End Program Repair with LLMs [2023-FSE]
22. Repair is nearly generation: Multilingual program repair with llms [2023-AAAI]
23. An empirical study of deep transfer learning-based program repair for Kotlin projects [2022-FSE]
24. Towards javascript program repair with generative pre-trained transformer (gpt-2) [2022-ICSE]
25. Invalidator: Automated patch correctness assessment via semantic and syntactic reasoning [2023-TSE]
26. A Unified Debugging Approach via LLM-Based Multi-Agent Synergy [2024-arXiv]
27. Hybrid Automated Program Repair by Combining Large Language Models and Program Analysis [2024-arXiv]
28. On the Reliability and Explainability of Language Models for Program Generation [2024-TOSEM]
29. DEAR A Novel Deep Learning-based Approach for Automated Program Repair [2022-ICSE]
30. Enhancing Automated Program Repair through Fine-tuning and Prompt Engineering [2023-arXiv]
31. Domain Knowledge Matters: Improving Prompts with Fix Templates for Repairing Python Type Errors [2024-ICSE]
32. A Novel Approach for Automatic Program Repair using Round-Trip Translation with Large Language Models [2024-arXiv]
33. Repairllama: Efficient representations and fine-tuned adapters for program repair [2023-arXiv]
34. An Analysis of the Automatic Bug Fixing Performance of ChatGPT [2023-APR]
35. LLM as Runtime Error Handler: A Promising Pathway to Adaptive Self-Healing of Software Systems [2024-arXiv]
36. Frustrated with Code Quality Issues? LLMs can Help! [2023-arXiv]
37. CORE: Resolving Code Quality Issues Using LLMs [2024-FSE]
38. Revisiting Evolutionary Program Repair via Code Language Model [2024-arXiv]
39. RAP-Gen: Retrieval-Augmented Patch Generation with CodeT5 for Automatic Program Repair [2023-FSE]
40. Copiloting the Copilots: Fusing Large Language Models with Completion Engines for Automated Program Repair [2023-FSE]
41. A prompt pattern catalog to enhance prompt engineering with chatgpt [2023-arXiv]
42. Addressing Compiler Errors: Stack Overflow or Large Language Models? [2023-arXiv]
43. Practical program repair in the era of large pre-trained language models [2022-arXiv]
44. Conversational automated program repair [2023-arXiv]
45. How Far Can We Go with Practical Function-Level Program Repair? [2024-arXiv]
46. Less Training, More Repairing Please: Revisiting Automated Program Repair via Zero-shot Learning [2022-FSE]
47. Automated Program Repair in the Era of Large Pre-trained Language Models [2023-ICSE]
48. Keep the Conversation Going- Fixing 162 out of 337 bugs for $0.42 each using ChatGPT [2023-ISSTA]
49. The Plastic Surgery Hypothesis in the Era of Large Language Models [2023-ASE]
50. Detecting, Creating, Repairing, and Understanding Indivisible Multi-Hunk Bugs [2024-FSE]
51. Towards Practical and Useful Automated Program Repair for Debugging [2024-arXiv]
52. Guiding ChatGPT to Fix Web UI Tests via Explanation-Consistency Checking [2023-arXiv]
53. Aligning LLMs for FL-free Program Repair [2024-arXiv]
54. CREF: An LLM-based Conversational Software Repair Framework for Programming Tutors [2024-ISSTA]
55. Multi-Objective Fine-Tuning for Enhanced Program Repair with LLMs [2024-arXiv]
56. Revisiting Unnaturalness for Automated Program Repair in the Era of Large Language Models [2024-arXiv]
57. ThinkRepair: Self-Directed Automated Program Repair [2024-ISSTA]
58. CIRCLE: Continual repair across programming languages [2022-ISSTA]
59. Using pre-trained language models to resolve textual and semantic merge conflicts (experience paper) [2022-ISSTA]
60. Neural Program Repair with Program Dependence Analysis and Effective Filter Mechanism [2023-arXiv]
61. STEAM: Simulating the Interactive Behavior of Programmers for Automatic Bug Fixing [2023-arXiv]
62. PyDex: Repairing Bugs in Introductory Python Assignments using LLMs [2024-OOPSLA]
63. Gamma: Revisiting template-based automated program repair via mask prediction [2023-ASE]
64. Enhancing LLM-Based Automated Program Repair with Design Rationales [2024-arXiv]
65. RePair: Automated Program Repair with Process-based Feedback [2024-ACL]
66. ConDefects: A Complementary Dataset to Address the Data Leakage Concern for LLM-Based Fault Localization and Program Repair [2024-FSE]

#### Report Severity Prediction

1. Graph Neural Network vs. Large Language Model: A Comparative Analysis for Bug Report Priority and Severity Prediction [2024-PROMISE]
2. BERT based severity prediction of bug reports for the maintenance of mobile applications [2024-JSS]
3. Method-Level Bug Severity Prediction using Source Code Metrics and LLMs [2023-ISSRE]

#### Sentiment analysis

1. Achieving reliable sentiment analysis in the software engineering domain using bert [2020-ICSME]
2. An Empirical Evaluation of the Zero-Shot, Few-Shot, and Traditional Fine-Tuning Based Pretrained Language Models for Sentiment Analysis in Software Engineering [2024-IEEE Access]
3. Sentiment analysis for software engineering: How far can pre-trained transformer models go? [2020-ICSME]
4. Revisiting Sentiment Analysis for Software Engineering in the Era of Large Language Models [2023-TOSEM]

#### Tag Recommendation

1. PTM4Tag: sharpening tag recommendation of stack overflow posts with pre-trained models [2022-ICPC]

#### Technical Debt Management

1. Towards Automatically Addressing Self-Admitted Technical Debt: How Far Are We? [2023-ASE]

#### Test Update

1. Identify and Update Test Cases when Production Code Changes- A Transformer-based Approach [2023-ASE]
2. Augmenting LLMs to Repair Obsolete Test Cases with Static Collector and Neural Reranker [2024-ISSRE]
3. Automated Test Case Repair Using Language Models [2024-arXiv]

#### Traceability Link Recovery

1. Enhancing Traceability Link Recovery with Unlabeled Data [2022-ISSRE]

#### Vulnerability Repair

1. Vision transformer inspired automated vulnerability repair [2024-TOSEM]
2. VulRepair: A T5-Based Automated Software Vulnerability Repair [2022-FSE]
3. LLM-Powered Code Vulnerability Repair with Reinforcement Learning and Semantic Reward [2024-arXiv]
4. A Case Study of LLM for Automated Vulnerability Repair: Assessing Impact of Reasoning and Patch Validation Feedback [2024-AIWARE]
5. Examining Zero-Shot Vulnerability Repair with Large Language Models [2023-SP]
6. Reality Check: Assessing GPT-4 in Fixing Real-World Software Vulnerabilities [2024-EASE]
7. ZeroLeak: Using LLMs for Scalable and Cost Effective Side-Channel Patching [2023-arXiv]
8. NAVRepair: Node-type Aware C/C++ Code Vulnerability Repair [2024-arXiv]
9. How Effective Are Neural Networks for Fixing Security Vulnerabilities [2023-ISSTA]
10. Exploring the Limits of ChatGPT in Software Security Applications [2023-arXiv]
11. Evaluating Pre-trained Language Models for Repairing API Misuses [2023-TOSEM]
12. Pre-Trained Model-Based Automated Software Vulnerability Repair: How Far are We? [2023-TDSC]

#### Code Clone Detection

1. CCT-Code: Cross-Consistency Training for Multilingual Clone Detection and Code Search [2023-arXiv]


### 📈Software Management

#### Developers' Behavior Analysis

1. Uncovering the Causes of Emotions in Software Developer Communication Using Zero-shot LLMs [2024-ICSE]

#### Effort estimation

1. Evaluation of Context-Aware Language Models and Experts for Effort Estimation of Software Maintenance Issues [2022-ICSME]
2. Fine-SE: Integrating Semantic Features and Expert Features for Software Effort Estimation [2024-ICSE]

#### Software Repository Mining 

1. LLM-Based Chatbots for Mining Software Repositories: Challenges and Opportunities [2024-EASE]


#### Software tool configuration

1. Can LLMs Configure Software Tools [2023-arXiv]




## 🧩RQ3

### 📊Benchmark

1. MBPP: Program Synthesis with Large Language Models [2021-arXiv]
2. MultiPL-E: a scalable and polyglot approach to benchmarking neural code generation [2023-TSE]
3. HumanEval: Evaluating Large Language Models Trained on Code [2021-arXiv]
4. ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation [2024-ICSE]
5. UniTSyn: A Large-Scale Dataset Capable of Enhancing the Prowess of Large Language Models for Program Testing [2024-ISSTA]
6. APPS: Measuring Coding Challenge Competence With APPS [2021-NeurIPS]
7. Competition-Level Code Generation with AlphaCode [2023-Science]
8. CoIR: A Comprehensive Benchmark for Code Information Retrieval Models [2024-arXiv]
9. EvalPlus-Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation [2023-NeurIPS]
10. Codexglue: A machine learning benchmark dataset for code understanding and generation [2021-NeurIPS]
11. The Vault: A Comprehensive Multilingual Dataset for Advancing Code Understanding and Generation [2023-EMNLP]
12. CrossCodeBench: Benchmarking Cross-Task Generalization of Source Code Models [2023-arXiv]
13. DebugBench: Evaluating Debugging Capability of Large Language Models [2024-ACL]
14. CodeLL: A Lifelong Learning Dataset to Support the Co-Evolution of Data and Language Models of Code [2024-MSR]
15. Codereval: A benchmark of pragmatic code generation with generative pre-trained models [2024-ICSE]
16. A Critical Review of Large Language Model on Software Engineering: An Example from ChatGPT and Automated Program Repair [2023-arXiv]
17. CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Benchmarking on HumanEval-X [2023-KDD]
18. BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions [2023-arXiv]
19. On the Evaluation of Neural Code Translation: Taxonomy and Benchmark [2023-ASE]
20. Can Machines Read Coding Manuals Yet? -- A Benchmark for Building Better Language Models for Code Understanding [2022-AAAI]

### 🗜️Compressing&Distillation

1. FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU [2023-ICML]
2. Compressing Pre-trained Models of Code into 3 MB [2022-ASE]
3. Distilled GPT for Source Code Summarization [2024-AUSE]

### 📚Education

1. Can ChatGPT Pass An Introductory Level Functional Language Programming Course? [2023-arXiv]
2. Chatgpt and software testing education: Promises & perils [2023-ICST]
3. Is this Snippet Written by ChatGPT? An Empirical Study with a CodeBERT-Based Classifier [2023-arXiv]
4. Lost at C: A User Study on the Security Implications of Large Language Model Code Assistants [2023-USENIX Security]
5. Does ChatGPT Help With Introductory Programming? An Experiment of Students Using ChatGPT in CS1 [2024-ICSE@SEET]
6. Is ChatGPT the Ultimate Programming Assistant--How far is it? [2023-arXiv]

### 🧮Empirical

1. ChatGPT, be my teaching assistant! Automatic Correction of SQL Exercises [2024-COMPSAC]
2. Enhancing Programming Learning with LLMs: Prompt Engineering and Flipped Interaction [2024-ASSE]
3. Can GPT-4 Replicate Empirical Software Engineering Research? [2024-FSE]
4. LLM-based Test-driven Interactive Code Generation: User Study and Empirical Evaluation [2024-arXiv]
5. ChatGPT for Vulnerability Detection, Classification, and Repair: How Far Are We? [2023-APSEC]
6. What Makes Good In-context Demonstrations for Code Intelligence Tasks with LLMs? [2023-ASE]
7. The Promise and Challenges of using LLMs to Accelerate the Screening Process of Systematic Reviews [2024-EASE]
8. An empirical study of ChatGPT-3.5 on question answering and code maintenance [2023-arXiv]
9. "Will I be replaced?" Assessing ChatGPT's effect on software development and programmer perceptions of AI tools [2024-SCP]
10. An Exploratory Evaluation of Large Language Models Using Empirical Software Engineering Tasks [2024-Internetware]
11. Using Transfer Learning for Code-Related Tasks [2022-TSE]
12. An empirical comparison of pre-trained models of source code [2023-ICSE]
13. Detecting LLM-Generated Text in Computing Education: Comparative Study for ChatGPT Cases [2024-COMPSAC]
14. LLM4TDD: Best Practices for Test Driven Development Using Large Language Models [2024-ICSE@LLM4Code]
15. An Empirical Study on Usage and Perceptions of LLMs in a Software Engineering Project [2024-ICSE@LLM4Code]
16. Utilization of pre-trained language models for adapter-based knowledge transfer in software engineering [2024-EMSE]
17. Extending the Frontier of ChatGPT: Code Generation and Debugging [2023-arXiv]
18. ChatGPT-Resistant Screening Instrument for Identifying Non-Programmers [2024-ICSE]
19. AI to the Test: Measuring ChatGPT’s Objective Accuracy in the SATs in Comparison to Human Performance [2024-COMPSAC]
20. ChatGPT Incorrectness Detection in Software Reviews [2024-ICSE]
21. Unveiling ChatGPT's Usage in Open Source Projects: A Mining-based Study [2024-MSR]
22. What do they capture? a structural analysis of pre-trained language models for source code [2022-ICSE]
23. Characterizing Developers’ Behaviors in LLM-Supported Software Development [2024-COMPSAC]
24. Rocks Coding, Not Development–A Human-Centric, Experimental Evaluation of LLM-Supported SE Tasks [2024-FSE]
25. Chatgpt prompt patterns for improving code quality, refactoring, requirements elicitation, and software design [2024-Generative AI]
26. The Devil is in the Tails: How Long-Tailed Code Distributions Impact Large Language Models [2023-ASE]
27. Are Large Language Models a Threat to Programming Platforms? An Exploratory Study [2024-ESEM]
28. ChatGPT application in Systematic Literature Reviews in Software Engineering: an evaluation of its accuracy to support the selection activity [2024-ESEM]
29. Optimizing the Utilization of Large Language Models via Schedule Optimization: An Exploratory Study [2024-ESEM]
30. An extensive study on pre-trained models for program understanding and generation [2022-ISSTA]
31. ChatGPT: A Study on its Utility for Ubiquitous Software Engineering Tasks [2023-arXiv]
32. Prompt Engineering or Fine Tuning: An Empirical Assessment of Large Language Models in Automated Software Engineering Tasks [2023-arXiv]
33. Using an LLM to Help With Code Understanding [2024-ICSE]
34. Few-shot learning for sentence pair classification and its applications in software engineering [2023-arXiv]

### 🎛️Tuning

1. Learning in the Wild: Towards Leveraging Unlabeled Data for Effectively Tuning Pre-trained Code Models [2024-ICSE]
2. Keeping Pace with Ever-Increasing Data: Towards Continual Learning of Code Intelligence Models [2023-ICSE]
3. An Empirical Study of Parameter-Efficient Fine-Tuning Methods for Pre-trained Code Models [2023-ASE]
4. Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large Language Models [2023-arXiv]
5. On the Usage of Continual Learning for Out-of-Distribution Generalization in Pre-trained Language Models of Code [2023-FSE]
6. CIRCLE: Continual repair across programming languages[2023-ISSTA]
7. Astraios: Parameter-Efficient Instruction Tuning Code Large Language Models [2024-arXiv]
8. Towards Efficient Fine-tuning of Pre-trained Code Models: An Experimental Study and Beyond [2023-ISSTA]
9. One Adapter for All Programming Languages? Adapter Tuning for Code Search and Summarization [2023-ICSE]

