# TTS Reference Validator — 사용 가이드

> Zero-shot 음색 클로닝 TTS에서 **"안정적인 reference audio"를 자동으로 선별**하는 2-스테이지 검증 파이프라인.
> 평가 단위는 합성본 개별이 아니라 **reference audio** 입니다. 학습은 하지 않습니다(추론·검증 전용).

---

## 목차

1. [개요](#개요)
2. [설치](#설치)
3. [빠른 시작](#빠른-시작)
4. [Manifest 형식](#manifest-형식)
5. [Stage 1 — 1:1 빠른 필터](#stage-1--11-빠른-필터)
6. [Stage 2 — 1:다 점수화 & 랭킹](#stage-2--1다-점수화--랭킹)
7. [출력 파일](#출력-파일)
8. [지표 ↔ 함수 매핑](#지표--함수-매핑)
9. [설정 & 튜닝](#설정--튜닝)
10. [주의사항 & FAQ](#주의사항--faq)

---

## 개요

| | Stage 1 | Stage 2 |
|---|---|---|
| **목적** | 명백히 나쁜 reference를 싸게 거르기 | 통과 reference를 정밀 점수화·랭킹 |
| **합성 구성** | ref 1개 ↔ 합성 1개 (1:1) | ref 1개 ↔ 여러 합성 (1:다) |
| **사용 지표** | F0 / Energy 내재 안정성 + 운율 정합 + ref 사전품질 | 위 전부 + SECS · UTMOS · CER · 국소 발화속도 |
| **무거운 모델** | 사용 안 함 (빠름) | ECAPA · WavLM · UTMOSv2 · Whisper |
| **결과** | 통과 ref 목록 | 1~N위 랭킹 + 항목별 점수 |

> **핵심 아이디어** — Stage 1에서 F0·Energy만으로 빠르게 거른 뒤, 살아남은 reference에만 비싼 전체 평가(Stage 2)를 적용합니다.

**입력 SR 처리** — 입력이 24 kHz여도 분석은 16 kHz로 자동 리샘플됩니다 (torchcrepe · ECAPA · WavLM-SV가 16 kHz를 요구).

---

## 설치

```bash
pip install -r requirements.txt
# UTMOSv2 는 git 설치 권장
pip install git+https://github.com/sarulab-speech/UTMOSv2.git
```

> 권장 환경: conda env `flow` · Python 3.10+ · CUDA 12.x · A6000.
> 모든 무거운 모델은 lazy-load + 캐시되며, 미설치 시 해당 지표만 `NaN`으로 비우고 진행합니다.

---

## 빠른 시작

```bash
# 1) Stage 1 : ref 1개당 1합성으로 빠르게 필터
python tts_reference_validator.py --stage 1 \
    --manifest s1.csv --out stage1.csv --device cuda

# 2) 통과한 ref(stage1_passed_refs.csv)에만 여러 텍스트를 합성해 s2.csv 구성 후
python tts_reference_validator.py --stage 2 \
    --manifest s2.csv --out stage2.csv --device cuda --top 10 --per-metric

# 3) 이후 순위(예: 11~30위)·항목별 점수를 재계산 없이 다시 조회
python tts_reference_validator.py --show stage2_ranking.csv \
    --rank-start 11 --rank-end 30 --per-metric
```

---

## Manifest 형식

CSV, UTF-8. 컬럼은 다음과 같습니다.

| 컬럼 | 필수 | 설명 |
|---|:---:|---|
| `ref_path` | O | reference 오디오 경로 |
| `synth_path` | O | 해당 reference로 합성한 오디오 경로 |
| `target_text` | O | 합성에 사용한 한국어 텍스트(정답 텍스트) |
| `speaker_id` | - | 화자 ID (그룹별 동적 임계용) |
| `dataset` | - | 데이터셋/소스 (그룹별 동적 임계용) |

- **Stage 1** : reference 1개당 **1행**.
- **Stage 2** : reference 1개당 **여러 행**(여러 합성).

```csv
ref_path,synth_path,target_text,speaker_id,dataset
/data/ref/spk01_a.wav,/synth/spk01_a_t1.wav,오늘 날씨가 좋네요,spk01,AIHUB
/data/ref/spk01_b.wav,/synth/spk01_b_t1.wav,내일은 비가 온대요,spk01,AIHUB
```

---

## Stage 1 — 1:1 빠른 필터

reference와 합성본을 1:1로 비교해, **합성이 튀거나 불안정한 reference를 먼저 제거**합니다.

### 평가 지표

| 지표 | 의미 | 방향 |
|---|---|:---:|
| `f0_spike_rate` | 유성 구간 F0 스파이크 비율 (Hampel) | 낮을수록 좋음 |
| `f0_octave_jumps` | 인접 유성 프레임 옥타브 점프 횟수 | 낮을수록 좋음 |
| `energy_burst_rate` | 급격한 음량 버스트/드롭아웃 비율 | 낮을수록 좋음 |
| `vuv_flap_per_sec` | 초당 유성↔무성 토글 (떨림·갈라짐) | 낮을수록 좋음 |
| `long_pause_sec` | 발화 중간 최장 묵음 길이 | 낮을수록 좋음 |
| `f0_register_delta_st` | ref 대비 음역(중앙값) 차이 (semitone) | 낮을수록 좋음 |
| `loudness_delta_db` | ref 대비 라우드니스 차이 (dB) | 낮을수록 좋음 |
| `speaking_rate_cps` | 글자/초 — 발화속도 이상 탐지 | 동적 임계(median±k·MAD) |
| `ref_prefilter_ok` | reference 자체 품질(길이·클리핑 등) | 통과/탈락 |

### 통과 조건

위 하드 게이트를 **모두** 통과하고 발화속도가 동적 범위 안에 들면 `PASS`.
출력된 `*_passed_refs.csv`의 reference만 Stage 2로 넘깁니다.

---

## Stage 2 — 1:다 점수화 & 랭킹

통과한 reference 1개당 여러 텍스트를 합성한 뒤, 7개 항목을 **배치 내 백분위(0~100점)** 로 환산해 종합 순위를 만듭니다.

### 점수 항목 (7개)

| 항목 | 구성 raw 지표 | 가중치 |
|---|---|:---:|
| 음색유사도 (SECS) | `secs` = ECAPA · WavLM 평균 | 0.25 |
| 자연스러움 (UTMOS) | `utmos` (UTMOSv2) | 0.20 |
| 명료도 (CER) | `cer` (Whisper 기반) | 0.18 |
| F0 안정성 | `f0_spike_rate`, `f0_octave_jumps` | 0.13 |
| 에너지 안정성 | `energy_burst_rate` | 0.10 |
| 타이밍 안정성 | `vuv_flap_per_sec`, `long_pause_sec`, `slow_word_rate` | 0.09 |
| 운율 정합 | `f0_register_delta_st`, `loudness_delta_db` | 0.05 |

> **점수화 방식** — "낮을수록 좋은" 지표는 방향을 뒤집은 뒤, reference별 평균에 대해 **백분위 순위(0~100)** 를 매깁니다. `composite`는 항목 점수의 가중 평균(0~100)이며, 이 값으로 1위부터 정렬합니다.
> 모두 **상대 점수**이므로 "이 배치 안에서 상위 몇 %"로 해석하세요.

### 국소 발화속도(느린 단어) 탐지

명료도(CER) 계산에 쓰는 Whisper의 **단어 타임스탬프를 재활용**해, 단어별 글자/초(CPS)를 구합니다.
발화 중앙값 대비 과도하게 느리거나(`slow_word_frac`), 짧은 단어가 비정상적으로 길게 끌리면(`slow_word_abs_dur`) `slow_word_rate`에 반영됩니다. 추가 비용이 거의 없습니다.

### 랭킹 출력 예시

```
=== Reference 랭킹  1~10위 ===

#  1  spk01_b.wav   composite=99.2  (n=3)
        음색유사도(SECS)      ########## 100.0
        자연스러움(UTMOS)     ########## 100.0
        명료도(CER)         ########## 100.0
        F0안정성            #########.  93.8
        에너지안정성           ########## 100.0
        타이밍안정성           ########## 100.0
        운율정합             ########## 100.0
```

- 기본은 **상위 10위 + 항목별 점수**.
- `--show` 모드로 **이후 순위**(예: 11~30위)나 다른 범위를 재계산 없이 다시 볼 수 있습니다.

---

## 출력 파일

| 파일 | 내용 |
|---|---|
| `stage1.csv` | Stage 1 쌍별 전체 지표 + 게이트 통과 여부 |
| `stage1_passed_refs.csv` | Stage 1 통과 reference 목록 |
| `stage2.csv` | Stage 2 쌍별 raw 지표 (모든 합성본) |
| `stage2_ranking.csv` | reference별 랭킹 · 항목 점수 · raw 평균 |

---

## 지표 ↔ 함수 매핑

| 개념 | 함수 |
|---|---|
| F0 추출 (torchcrepe → pyworld 폴백) | `extract_f0` |
| F0 스파이크 · 옥타브 점프 | `f0_spike_metrics` |
| 에너지 버스트 / 드롭아웃 | `energy_burst_metrics` |
| V/UV flap · 긴 묵음 | `vuv_and_pause_metrics` |
| ref↔synth 운율 정합 | `prosody_match` |
| reference 사전 품질 | `reference_quality` |
| 음색 유사도 (SECS) | `secs_scores` |
| 자연스러움 (UTMOSv2) | `utmos_score` |
| 명료도 (CER) + 단어 타임스탬프 | `asr_analyze`, `cer` |
| 국소 발화속도(느린 단어) | `flag_slow_words` |
| Stage 2 항목 백분위 점수 | `_pct_subscores` |

---

## 설정 & 튜닝

모든 임계값은 파일 상단 `Config` 데이터클래스에 모여 있습니다. 자주 만지는 값:

| 파라미터 | 기본값 | 설명 |
|---|:---:|---|
| `sr` | 16000 | 분석 SR (입력 24k는 자동 리샘플) |
| `crepe_conf_thr` | 0.50 | 유성 판정 confidence 하한 |
| `hampel_min_abs_f0` | 2.0 | F0 스파이크 절대 하한 (semitone) |
| `hampel_min_abs_energy` | 6.0 | 에너지 버스트 절대 하한 (dB) |
| `max_f0_spike_rate` | 0.03 | Stage 1 F0 스파이크 게이트 |
| `max_long_pause_sec` | 1.2 | Stage 1 긴 묵음 게이트 |
| `speaking_rate_mad_k` | 3.0 | 발화속도 동적 임계 폭 |
| `slow_word_frac` | 0.40 | 느린 단어 판정(중앙 CPS 대비) |
| `ITEM_WEIGHTS` | — | Stage 2 항목 가중치 |

> **권장 워크플로** — 처음엔 게이트를 느슨하게(`max_f0_spike_rate=0.05`, `max_long_pause_sec=1.5`) 두고 통과/탈락 분포를 확인한 뒤 점진적으로 조이세요.

---

## 주의사항 & FAQ

**Q. UTMOSv2 점수를 그대로 믿어도 되나요?**
UTMOSv2는 영어 MOS로 학습돼 **한국어 절대 점수는 보정되어 있지 않습니다.** 본 파이프라인은 배치 내 **상대 백분위**로만 사용합니다. 한국어 human-rated 샘플 20~30개로 UTMOS와의 순위상관(Spearman)을 한 번 확인하고, 낮으면 `ITEM_WEIGHTS`에서 자연스러움 비중을 줄이고 F0/CER 비중을 올리세요.

**Q. WER이 아니라 CER을 쓰는 이유는?**
한국어는 띄어쓰기(어절 분절)가 불안정해 WER이 spacing 오류로 부풀려집니다. 음절 단위 CER이 더 안정적이며, 더 미세하게 보려면 `cer_jamo`(자모 단위)도 함께 출력됩니다.

**Q. reference가 노이즈가 있는데 F0가 자꾸 튀어요.**
CREPE는 클린 음성에 강하지만 잡음엔 RMVPE가 더 강건합니다. `extract_f0`를 RMVPE 호출로 교체하면 됩니다(나머지 코드는 그대로). 어떤 추출기를 쓰든 **모든 합성본에 동일 추출기·동일 파라미터**를 적용하는 일관성이 가장 중요합니다.

**Q. 점수가 낮을수록 좋은 지표와 높을수록 좋은 지표가 헷갈려요.**
"이상치·거리·오류" 계열(스파이크율, 버스트율, CER, register delta 등)은 **낮을수록** 좋고, "유사도·품질" 계열(SECS, UTMOS)은 **높을수록** 좋습니다. Stage 2 항목 점수는 이 방향을 모두 통일해 **높을수록 좋은 0~100점**으로 환산합니다.
