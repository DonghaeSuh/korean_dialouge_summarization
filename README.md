# 일상 대화 요약
본 리포지토리는 '2024년 국립국어원 인공지능의 한국어 능력 평가' 경진 대회 과제 중 '일상 대화 요약'에 대한 모델의 학습과 평가를 재현하기 위한 코드를 포함하고 있습니다.

<br/>

위 실험에 대한 자세한 설명은 다음 블로그 글[[링크](https://blog.naver.com/gypsi12/223569538573)]에서 보실 수 있습니다.

## Table of contents
1. [Summary](#summary)
3. [Experimental results](#experimental-results)
4. [Instructions](#instructions)
5. [Usage instructions](#usage-instructions)
7. [References](#references)

## Summary
일상 대화 데이터에서 요약문을 생성해내는 Task입니다. (Dialogue Summarization)

이번 경진대회에서 새롭게 제안한 두 가지 기법은 다음과 같습니다.

- **음성 전사 과정에서 생기는 반복 어구(단어) 노이즈를 효과적으로 전처리하는 기법**
- **출력(output) 형식을 통일하여 데이터 효율적인 LLM fine-tuning 기법**

이를 통해 기존의 ROGUE와 BERTScore의 단점을 보완하고 장점을 결합한 BLEURT 점수에서 특히 주목할 만한 성과를 거두었습니다. 

또한, Inference 시 Greedy Decoding 기법을 사용함으로써, beam search와 같은 추가적인 메모리 오버헤드 없이 뛰어난 재현 능력을 발휘할 수 있었습니다. 이 모든 요소들이 결합되어, 높은 효율성과 품질을 동시에 실현하고 있습니다.

<br/>

## Experimental results

총 8개의 모델에 대한 리더보드 상 점수입니다.
- baseline : 대회 baseline
- baseline (with SFT) : 대회 baseline + SFT fine-tuning
- **TG** : 전반적인 요약(**T**otal Summary) 부분만 형식 통일(**G**eneralization) + Instruction Fine-tuning
- **TG + SG** : TG + 화자1, 2 요약(**S**peaker Summary) 형식 통일(**G**eneralization) 
- **TG + SG + RR** : TG + SG + 반복 어구(단어) 제거 전처리(**R**emove **R**epeated phrase and words)
- **TG + SG + RR + Total** : **TG + SG + RR** 세팅으로 Train+Dev 데이터 전부(**T**otal) 사용하여 학습
- **Ensemble_v1** : TG + SG + RR와 TG + SG + RR + Total 버전 중에서 짧은 문장을 선택
- **Ensemble_v2** : Ensemble_v1에서 화자 2 요약 부분만 생성하도록 따로 학습된 모델의 결과로 대체

<br/>

| Model | Evaluation Score | ROUGE-1 | BERTScore | BLEURT |
| --- | --- | --- | --- | --- |
| basleine | 54.276 |44.592 | 73.277 | 44.950 |
| baseline (with SFT) | 58.982 | **54.759** | **79.154** | 43.035 |
| **TG** | 59.148181 | 53.729551 | 78.3912502 | 45.3237418 |
| **TG + SG** | 59.6248691 | 53.4625498 | 78.5050607 | 46.9069969 |
| **TG + SG + RR** | 59.8890253 | 53.8221605 | 78.5433122 | 47.3016033 |
| **TG + SG + RR + Total**  | 59.8531238 | 54.1609176 | 78.535594 | 46.8628598 |
| **Ensemble_v1** | 60.0270675 | 53.7108599 | 78.5558022 | **47.8145403** |
| **Ensemble_v2** | **60.0316536** | 53.7539715 | 78.5728804 | 47.7681089 |


<br/>

## Instructions

### Hardware
사용한 하드웨어는 다음과 같습니다
- **Google Colab**
    - Intel(R) Xeon(R) 8-Core Processor 2.20GHz
    - NVIDIA® A100 , 40GB 


<br/>

### Conda environment setup
Mac, Windows, Linux에서 [conda](https://docs.continuum.io/free/anaconda/install/)를 설치합니다


1. **conda를 업데이트합니다**

   ```
   conda update conda
   ```

2. **레포지토리를 clone합니다.**

    ```
    git clone https://github.com/DonghaeSuh/korean_dialouge_summarization.git
    ```


3. **setup.sh 파일을 실행합니다**
    ```
    source setup.sh
    ```
    
    위 스크립트를 실행하면
    - korean_dialogue_summarization라는 이름의 **가상환경을 생성하고 활성화**한 이후

    - **필요 의존성 패키지들이 설치**되고
    - checkpoints라는 폴더에 3가지 chekcpoint 폴더가 생깁니다
        - **checkpoint-85** : "전반적인 요약" + "speaker 2 요약" 만 따로 학습한 모델입니다
        - **checkpoint-115** : **TG + SG + RR** 버전 모델입니다
        - **checkpoint-150** : **TG + SG + RR + Total** 버전 모델입니다
    - checkpoint가 존재하는 drive [링크](https://drive.google.com/drive/folders/1o4MNcMKMMb84Hn3YWHwF6jLNAsOJKdyR)입니다.

<br/>

### Code structure
```
├─ README.md
├─ configs
│  └─ tg_sg_rr_config.json
├─ eda
│  ├─ dot_with_space.ipynb
│  ├─ error_sample_eda.ipynb
│  ├─ length_eda.ipynb
│  ├─ name_token_eda.ipynb
│  ├─ name_token_eda_2.ipynb
│  ├─ output_structure_eda.ipynb
│  ├─ output_structure_eda_2.ipynb
│  ├─ output_structure_generalization.ipynb
│  ├─ preprocess.ipynb
│  ├─ repeat_eda.ipynb
│  ├─ repeat_output_eda_2.ipynb
│  ├─ repeat_utterance_eda.ipynb
│  ├─ repeat_weired_word.ipynb
│  ├─ stopword_eda.ipynb
│  ├─ utterance_count_eda.ipynb
│  ├─ utterance_count_eda_2.ipynb
│  └─ utterance_length_eda.ipynb
├─ resource
│  └─ data
│     ├─ 일상대화요약_dev.json
│     ├─ 일상대화요약_test.json
│     └─ 일상대화요약_train.json
├─ results
│  └─ test
│     ├─ TG_SG_RR_Total_result.json
│     ├─ TG_SG_RR_Total_result_postprocessed.json
│     ├─ TG_SG_RR_only_speaker_2_result.json
│     ├─ TG_SG_RR_only_speaker_2_result_postprocessed.json
│     ├─ TG_SG_RR_result.json
│     ├─ TG_SG_RR_result_postprocessed.json
│     ├─ ensemble_1.json
│     └─ ensemble_2.json
├─ run
│  ├─ __init__.py
│  ├─ test.py
│  └─ train_qlora.py
├─ make_ensemble_2.sh
├─ postprocess.py
├─ requirements.txt
├─ run_fast_inference.sh
├─ setup.sh
└─ src
   ├─ __init__.py
   ├─ data
   │  ├─ dev_repeated_phrase_indices_0.pkl
   │  ├─ dev_repeated_phrase_indices_1.pkl
   │  ├─ test_repeated_phrase_indices_0.pkl
   │  ├─ test_repeated_phrase_indices_1.pkl
   │  ├─ train_repeated_phrase_indices_0.pkl
   │  └─ train_repeated_phrase_indices_1.pkl
   ├─ data.py
   └─ utils.py

```

<br/>

## Usage instructions

### Training

1. **Wandb Login**

   ```
   wandb login
   ``` 
    - wandb를 통한 train loss와 dev evaluation 결과를 추적하기 위해 로그인합니다
    - 원하지 않는다면, 로그인을 하지 않고 넘어갑니다

<br/>

2. **config 설정**

    원하는 configuration 설정을 json 형태의 config 파일로 준비합니다

    ```
    {"seed": 42,

    "wandb": {
        "wandb_run_name" : "tg_sg_rr",
        "wandb_project_name": "korean_dialog",
        "wandb_entity_name": "gypsi12",
        "wandb_log_model": "checkpoint"},
    
    "arch": {
        "model_id": "MLP-KTLim/llama-3-Korean-Bllossom-8B",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "eval_accumulation_steps" : 4,
        "warmup_steps": 20,
        "lr" : 2e-5,
        "epoch": 25,
        "strategy": "steps",
        "steps": 5,
        "weight_decay": 0.1,
        "lr_scheduler_type": "cosine",
        "max_seq_length": 2048,
        "seed": 42,
        "metric_for_best_model": "loss",
        "early_stopping_patience": 10},

    "lora_arch": {
        "r": 16,
        "lora_alpha": 64,
        "lora_dropout": 0.1},

    "path": {
        "train_path": "resource/data/일상대화요약_train.json",
        "dev_path": "resource/data/일상대화요약_dev.json",
        "test_path": "resource/data/일상대화요약_test.json",
        "predict_path": "resource/data/일상대화요약_test.json",
        "chkpoint_save_dir": "resource/checkpoints"}
    }
    ```

<br/>

3. **학습**

    ```
    python -m run.train_qlora
    ```
    다음 코드를 실행한 이후 터미널에 다음과 같은 제시문이 뜨면
    ```
    ## input config path ##
    ```
    자신이 사용하고자 하는 config 파일을 확장자를 포함하여 전달하고 엔터(Enter)를 누릅니다
    ```
    예시) tg_sg_rr_config.json
    ```

<br/>

4. **체크포인트 관리**
    
    config내 "steps" 인자의 step마다 checkpoint가 "chkpoint_save_dir" 위치에 저장이 됩니다

<br/>

### Inference

1. **추론**

    추론하고자 하는 checkpoint의 경로를 adapter_checkpoint_path에 전달하고, 저장할 경로, 사용하는 모델 id(허깅페이스)를 전달해주고 실행합니다
    ```
    python -m run.test \
        --output {저장할 경로.json} \
        --model_id {사용하는 모델 id} \
        --device cuda:0 \
        --adapter_checkpoint_path {체크포인트 경로}
    ```

<br/>

2. **후처리**
    
    본 레포지토리의 방법론을 사용할 시, output 내에 (## 전반적인 요약, ## speaker 1 요약, ## speaker 2 요약)과 같은 Header 들이 달리게 됩니다.
    
    ```
    python -m postprocess \
    --path "results/TG_SG_RR_Total_result.json" \
    --output_path "results/TG_SG_RR_Total_result_postprocessed.json"
    ```
    이를 없에주기 위해 위 코드를 통해 postprocess.py 모듈로 후처리할 수 있습니다

<br/>

3. **앙상블**

    앙상블(더 짧은 문장을 선택)을 하기 위해서는
    후처리된 두 개의 json파일을 results 폴더에 마련한 이후

    ```
    python -m postprocess \
    --path "results/TG_SG_RR_result_postprocessed.json" \
    --ensemble_path_1 "results/TG_SG_RR_Total_result_postprocessed.json" \
    --output_path "results/ensemble_1.json"
    ```
    위 코드를 통해 ensemble_1 을 얻어볼 수 있습니다


<br/>

### Fast inference
**TG + SG + RR** 버전과, **TG + SG + RR + Total** 버전에 해당하는 checkpoint를 **setup.sh**를 통해 불러왔다면

```
source run_fast_inference.sh
```
위 스크립트를 통해 한번에 아래의 총 5가지 결과물을 results 폴더에서 확인하실 수 있습니다.
- ensemble_1.json
- TG_SG_RR_result_postprocessed.json :
- TG_SG_RR_result.json
- TG_SG_RR_Total_result_postprocessed.json
- TG_SG_RR_Total_result.json

<br/>

### Fast ensemble_2
최고 점수의 모델인 ensemble_2를 재현하기 위해서는 먼저 ensemble_1이 필요하고\
이후, 총 3가지 단계를 거쳐야 합니다
```
# checkpoint-85 모델로 test.py 실행
python -m run.test \
    --output "results/TG_SG_RR_only_speaker_2_result.json" \
    --model_id "MLP-KTLim/llama-3-Korean-Bllossom-8B" \
    --tokenizer "MLP-KTLim/llama-3-Korean-Bllossom-8B" \
    --device "cuda:0" \
    --adapter_checkpoint_path "checkpoints/checkpoint-85" \
    --is_test "yes" \
    --only_speaker_2 "yes"

# TG_SG_RR_only_speaker_2_result.json에 대해 후처리
python -m postprocess \
    --path "results/TG_SG_RR_only_speaker_2_result.json" \
    --output_path "results/TG_SG_RR_only_speaker_2_result_postprocessed.json"


# 두 후처리 결과 파일을 앙상블
python -m postprocess \
    --path "results/ensemble_1.json" \
    --ensemble_path_2 "results/TG_SG_RR_only_speaker_2_result_postprocessed.json" \
    --output_path "results/ensemble_2.json"
```
이를 한 번에 수행할 수 있는 스크립트는 아래와 같습니다
```
source make_ensemble_2.sh
```

<br/>


## References

huggingface/transformers (https://github.com/huggingface/transformers)  
Bllossome (Teddysum) (https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)  
국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
