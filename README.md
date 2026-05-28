# TTS Reference Validator

> Zero-shot 음색 클로닝 TTS에서 **안정적인 reference audio를 자동 선별**하는 2-스테이지 검증 파이프라인

`Python 3.10+` · `PyTorch` · `한국어 음성 기준` · 추론·검증 전용(학습 없음)

---

## 한눈에 보기

reference audio는 많지만(AIHUB 등), 같은 화자라도 발화마다 prosody·energy가 달라
**어떤 오디오를 reference로 넣느냐에 따라 합성이 자연스럽기도, 튀기도** 합니다.

이 파이프라인은 reference별로 합성 결과를 자동 채점해, **"넣으면 안정적으로 잘 나오는 reference"** 를 골라냅니다.
평가 단위는 합성본 개별이 아니라 **reference audio** 입니다.

```
                 ┌─────────────────────────────┐
  reference ×N   │  Stage 1  (1:1, 빠른 필터)   │   F0 · Energy · 운율정합 · ref품질
  ──────────────▶│  명백히 나쁜 ref 제거        │──┐  (무거운 모델 없음)
                 └─────────────────────────────┘  │
                                                   ▼  통과한 ref만
                 ┌─────────────────────────────┐
  ref당 여러 합성│  Stage 2  (1:다, 정밀 평가)  │   + SECS · UTMOS · CER · 국소 발화속도
  ──────────────▶│  7개 항목 점수화 → 랭킹      │──▶  1~N위 + 항목별 점수
                 └─────────────────────────────┘
```

---

## 주요 기능

- **2-스테이지 구조** — 싼 F0/Energy 필터(Stage 1)로 거른 뒤, 살아남은 reference에만 비싼 전체 평가(Stage 2) 적용
- **합성 글리치 탐지** — F0 스파이크·옥타브 점프, 에너지 버스트/드롭아웃, V/UV flap, 긴 묵음 (Hampel + 절대 하한)
- **타이밍 이상 탐지** — 발화속도(동적 임계) + 국소 "느린 단어"(Whisper 단어 타임스탬프 재활용)
- **음색·자연스러움·명료도** — SECS(ECAPA + WavLM-SV), UTMOSv2(상대 점수), 한국어 CER(+ jamo-CER)
- **백분위 랭킹** — 7개 항목을 0~100점으로 환산해 reference 1~N위, 항목별 점수까지 확인
- **24 kHz 입력 자동 처리** — 분석은 16 kHz로 자동 리샘플

---

## 설치

```bash
pip install -r requirements.txt
pip install git+https://github.com/sarulab-speech/UTMOSv2.git   # UTMOSv2
```

> 권장: conda env `flow` · Python 3.10+ · CUDA 12.x · A6000

---

## 빠른 시작

```bash
# Stage 1 : ref 1개당 1합성으로 빠르게 필터
python tts_reference_validator.py --stage 1 --manifest s1.csv --out stage1.csv --device cuda

# Stage 2 : 통과한 ref에 여러 텍스트를 합성해 점수화·랭킹 (top 10 + 항목별 점수)
python tts_reference_validator.py --stage 2 --manifest s2.csv --out stage2.csv --device cuda --top 10 --per-metric

# 이후 순위(예: 11~30위) 재조회 (재계산 없이 CSV에서)
python tts_reference_validator.py --show stage2_ranking.csv --rank-start 11 --rank-end 30 --per-metric
```

manifest 컬럼: `ref_path, synth_path, target_text [, speaker_id, dataset]`
(Stage 1은 ref당 1행, Stage 2는 ref당 여러 행)

---

## 파일 구성

```
.
├── tts_reference_validator.py   # 메인 파이프라인 (Stage 1 / Stage 2 / 랭킹)
├── requirements.txt             # 의존성
├── USAGE.md                     # 상세 사용 가이드
└── README.md                    # (이 문서)
```

---

## 출력

| 파일 | 내용 |
|---|---|
| `stage1.csv` | Stage 1 쌍별 지표 + 게이트 통과 여부 |
| `stage1_passed_refs.csv` | Stage 1 통과 reference 목록 |
| `stage2.csv` | Stage 2 쌍별 raw 지표 |
| `stage2_ranking.csv` | reference별 랭킹 · 항목 점수 · raw 평균 |

랭킹 출력 예시:

```
#  1  spk01_b.wav   composite=99.2  (n=3)
        음색유사도(SECS)      ########## 100.0
        자연스러움(UTMOS)     ########## 100.0
        명료도(CER)         ########## 100.0
        F0안정성            #########.  93.8
        ...
```

---

## 자세한 사용법

각 지표 정의, 점수화 방식, `Config` 튜닝, FAQ는 **[USAGE.md](./USAGE.md)** 를 참고하세요.

---

## 참고 / 주의

- **UTMOSv2** 는 영어 MOS로 학습돼 한국어 절대 점수는 보정되어 있지 않습니다. 배치 내 **상대 점수**로만 사용하며, 한국어 샘플로 순위상관을 한 번 점검하는 것을 권장합니다.
- F0 추출은 `torchcrepe` 기본, 노이즈가 많은 reference엔 `extract_f0`를 RMVPE로 교체하세요. 어떤 추출기든 **모든 합성본에 동일 추출기·동일 파라미터** 적용이 핵심입니다.
