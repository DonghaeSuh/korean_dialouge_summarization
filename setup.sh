#!/bin/bash

# "korean_dialogue_summarization"이라는 가상환경 생성
# conda가 설치되어 있으면 conda를 사용, 없다면 venv를 사용
# Python 3.10.12 버전을 사용

if command -v conda &> /dev/null
then
    # conda로 가상환경 생성 (Python 3.10.12)
    conda create -n korean_dialogue_summarization python=3.10.12 -y
    # 가상환경 활성화
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate korean_dialogue_summarization
elif command -v python3 &> /dev/null && command -v python3 -m venv &> /dev/null
then
    # venv로 가상환경 생성 (Python 3.10.12이 이미 시스템에 설치되어 있어야 함)
    python3 -m venv korean_dialogue_summarization
    # 가상환경 활성화
    source korean_dialogue_summarization/bin/activate
    # 가상환경에서 pip 업그레이드
    pip install --upgrade pip
    # python 버전 확인 후 3.10.12가 아니면 경고
    PYTHON_VERSION=$(python --version 2>&1)
    if [[ $PYTHON_VERSION != "Python 3.10.12" ]]; then
        echo "경고: 현재 Python 버전은 3.10.12가 아닙니다. ($PYTHON_VERSION)"
    fi
else
    echo "conda 또는 venv가 설치되어 있지 않습니다. 먼저 설치해 주세요."
    exit 1
fi

# requirements.txt에 정의된 라이브러리 설치
pip install -r requirements.txt

export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc

# Google Drive에서 폴더 다운로드
gdown --folder https://drive.google.com/drive/folders/1o4MNcMKMMb84Hn3YWHwF6jLNAsOJKdyR

# checkpoints 디렉토리로 이동
cd checkpoints

# 압축 파일 목록
zip_files=("checkpoint-85.zip" "checkpoint-115.zip" "checkpoint-150.zip")

# 각 zip 파일을 압축 해제하고 원본 zip 파일 삭제
for zip_file in "${zip_files[@]}"
do
    if [ -f "$zip_file" ]; then
        unzip "$zip_file"
        rm "$zip_file"
    fi
done

# 가상환경은 계속 활성화된 상태로 유지
echo "작업이 완료되었습니다. 가상환경 'korean_dialogue_summarization'이 활성화된 상태로 유지됩니다."
echo "Python 버전: $(python --version)"

# 메인 폴더로 복귀
cd ..