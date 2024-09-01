#!/bin/bash

# 가상환경 활성화
if command -v conda &> /dev/null
then
    # conda 환경 활성화
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate korean_dialogue_summarization
elif command -v python3 &> /dev/null
then
    # venv 환경 활성화
    source korean_dialogue_summarization/bin/activate
else
    echo "가상환경이 활성화되어 있지 않습니다. 가상환경을 먼저 활성화해 주세요."
    exit 1
fi

# checkpoint-85 모델로 test.py 실행
python -m run.test \
    --output "$RESULTS_DIR/TG_SG_RR_only_speaker_2_result.json" \
    --model_id "MLP-KTLim/llama-3-Korean-Bllossom-8B" \
    --tokenizer "MLP-KTLim/llama-3-Korean-Bllossom-8B" \
    --device "cuda:0" \
    --adapter_checkpoint_path "checkpoints/checkpoint-85" \
    --is_test "yes" \
    --only_speaker_2 "yes"

# TG_SG_RR_only_speaker_2_result.json에 대해 후처리
python -m postprocess \
    --path "$RESULTS_DIR/TG_SG_RR_only_speaker_2_result.json" \
    --output_path "$RESULTS_DIR/TG_SG_RR_only_speaker_2_result_postprocessed.json"

echo "Inference 작업이 완료되었습니다."

# 두 후처리 결과 파일을 앙상블
python -m postprocess \
    --path "$RESULTS_DIR/ensemble_1.json" \
    --ensemble_path_2 "$RESULTS_DIR/TG_SG_RR_only_speaker_2_result_postprocessed.json" \
    --output_path "$RESULTS_DIR/ensemble_2.json"

echo "후처리 및 앙상블 작업이 완료되었습니다. 결과 파일이 $RESULTS_DIR/ensemble_2.json으로 저장되었습니다."
