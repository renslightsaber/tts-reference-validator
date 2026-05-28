#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tts_reference_validator.py  (2-stage)
=====================================
Zero-shot 음색 클로닝 TTS에서 "안정적인 reference audio"를 자동 선별하는 2-스테이지 파이프라인.

평가 단위 = reference audio.   학습 없음(추론/검증 전용).
입력 음성 SR = 24000 Hz 라고 가정하되, 분석은 16000 Hz 로 자동 리샘플
(torchcrepe / ECAPA / WavLM-SV 가 16kHz 를 요구하므로).

────────────────────────────────────────────────────────────────────────
Stage 1  (1:1, 싸고 빠른 필터)
    ref 1개  ↔  합성 1개.
    F0 / Energy 내재 안정성 + ref↔synth 운율 정합 + reference 사전 품질만.
    (ASR/UTMOS/SECS 안 돌림) → 통과한 ref 목록을 다음 단계로.
    지표: f0_spike_rate, f0_octave_jumps, energy_burst_rate,
          vuv_flap_per_sec, long_pause_sec, f0_register_delta_st,
          loudness_delta_db, speaking_rate_cps(동적), reference 사전품질

Stage 2  (1:다, 전체 평가 + 랭킹)
    통과한 ref 1개당 여러 target text 합성.
    전체 항목을 배치 내 '백분위(0~100)' 로 점수화 → ref별 집계 → composite 랭킹.
    항목: 음색유사도(SECS=ECAPA+WavLM), 자연스러움(UTMOS), 명료도(CER),
          F0안정성, 에너지안정성, 타이밍안정성(V/UV·긴묵음·느린단어), 운율정합
    출력: stageN_pairs.csv (쌍 단위) + stage2_ranking.csv (ref별 1~N위)
────────────────────────────────────────────────────────────────────────

사용 예
    # Stage 1 : 통과 ref 필터
    python tts_reference_validator.py --stage 1 --manifest s1.csv --out stage1.csv --device cuda
    # Stage 2 : 통과 ref 들에 대해 1:다 평가 + 랭킹(top 10, 항목별 점수)
    python tts_reference_validator.py --stage 2 --manifest s2.csv --out stage2.csv --device cuda --top 10 --per-metric
    # 11~30위를 나중에 다시 조회
    python tts_reference_validator.py --show stage2_ranking.csv --rank-start 11 --rank-end 30 --per-metric

manifest 컬럼:  ref_path, synth_path, target_text [, speaker_id, dataset]
    Stage1 : ref 1개당 1행.   Stage2 : ref 1개당 여러 행(여러 합성).
"""

from __future__ import annotations

import argparse
import os
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import soundfile as sf
from scipy import ndimage

try:
    import librosa
except Exception:
    librosa = None


# ───────────────────────────── 설정 ─────────────────────────────
@dataclass
class Config:
    input_sr: int = 24000              # 입력 파일 SR (참고용; 실제론 파일 헤더에서 읽음)
    sr: int = 16000                    # 분석 표준 SR (모델 호환). 24k→16k 자동 리샘플
    hop_ms: float = 10.0
    win_ms: float = 25.0

    # F0 / CREPE
    f0_min: float = 65.0
    f0_max: float = 500.0
    crepe_conf_thr: float = 0.50

    # Hampel (스파이크/버스트)
    hampel_win: int = 7
    hampel_nsigma_f0: float = 3.0
    hampel_nsigma_energy: float = 3.0
    hampel_min_abs_f0: float = 2.0     # semitone (매끈 곡선 오탐 방지)
    hampel_min_abs_energy: float = 6.0 # dB
    octave_jump_semitone: float = 11.0

    # 묵음 / 발화속도
    silence_rel_db: float = 40.0       # peak 대비 -40dB 이하 = 묵음
    slow_word_frac: float = 0.40       # 발화 중앙 CPS 의 40% 미만이면 '느림'
    slow_word_min_dur: float = 0.30    # 이 길이(초) 이상인 단어만 평가
    slow_word_abs_dur: float = 0.80    # 짧은 단어(≤2글자)가 이 길이 넘으면 드래그

    # Stage 1 하드 게이트
    max_f0_spike_rate: float = 0.03
    max_f0_octave_jumps: int = 1
    max_energy_burst_rate: float = 0.03
    max_vuv_flap_per_sec: float = 8.0
    max_long_pause_sec: float = 1.2
    speaking_rate_mad_k: float = 3.0   # median±k·MAD (동적)
    ref_min_dur: float = 3.0
    ref_max_dur: float = 30.0
    clip_thr: float = 0.99
    max_clip_rate: float = 1e-3

    # Stage 2 명료도 게이트(참고 플래그용)
    max_cer: float = 0.10
    utmos_keep_quantile: float = 0.25


CFG = Config()

# Stage 2 항목(점수화 그룹) 정의 — (raw 컬럼, 방향)
ITEM_GROUPS = {
    "음색유사도(SECS)":   [("secs", "higher")],
    "자연스러움(UTMOS)":  [("utmos", "higher")],
    "명료도(CER)":        [("cer", "lower")],
    "F0안정성":           [("f0_spike_rate", "lower"), ("f0_octave_jumps", "lower")],
    "에너지안정성":       [("energy_burst_rate", "lower")],
    "타이밍안정성":       [("vuv_flap_per_sec", "lower"), ("long_pause_sec", "lower"),
                           ("slow_word_rate", "lower")],
    "운율정합":           [("f0_register_delta_st", "lower"), ("loudness_delta_db", "lower")],
}
ITEM_WEIGHTS = {
    "음색유사도(SECS)": 0.25, "자연스러움(UTMOS)": 0.20, "명료도(CER)": 0.18,
    "F0안정성": 0.13, "에너지안정성": 0.10, "타이밍안정성": 0.09, "운율정합": 0.05,
}


# ───────────────────────────── 오디오 I/O ─────────────────────────────
def load_audio(path: str, sr: int = CFG.sr) -> np.ndarray:
    """mono float32, 분석 SR(기본 16k)로. 24k 등 어떤 SR 이든 자동 리샘플."""
    y, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return np.ascontiguousarray(y)


def frame_rms_db(y, sr, hop_ms, win_ms):
    hop = int(sr * hop_ms / 1000)
    win = int(sr * win_ms / 1000)
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True)[0]
    return 20.0 * np.log10(np.maximum(rms, 1e-8))


# ───────────────────────────── F0 ─────────────────────────────
def extract_f0(y, sr, cfg: Config = CFG, device="cpu"):
    """returns (f0_hz[T], confidence[T]).  무성/무신뢰 = 0."""
    hop = int(sr * cfg.hop_ms / 1000)
    try:
        import torch
        import torchcrepe
        wav = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
        f0, conf = torchcrepe.predict(
            wav, sr, hop_length=hop, fmin=cfg.f0_min, fmax=cfg.f0_max,
            model="full", batch_size=512, device=device, return_periodicity=True)
        f0 = f0.squeeze(0).cpu().numpy()
        conf = conf.squeeze(0).cpu().numpy()
        f0[conf < cfg.crepe_conf_thr] = 0.0
        return f0, conf
    except Exception:
        import pyworld as pw
        _f0, t = pw.harvest(y.astype(np.float64), sr, f0_floor=cfg.f0_min,
                            f0_ceil=cfg.f0_max, frame_period=cfg.hop_ms)
        f0 = pw.stonemask(y.astype(np.float64), _f0, t, sr).astype(np.float32)
        return f0, (f0 > 0).astype(np.float32)


def hz_to_semitone(f0):
    st = np.full_like(f0, np.nan, dtype=np.float64)
    v = f0 > 0
    st[v] = 12.0 * np.log2(f0[v])
    return st


# ───────────────────────────── Hampel & 안정성 지표 ─────────────────────────────
def hampel_outliers(x, win, nsigma, min_abs=0.0):
    """주변 median±n·MAD 를 벗어나고 동시에 |dev|>min_abs 인 프레임을 outlier 로."""
    out = np.zeros_like(x, dtype=bool)
    idx = np.where(~np.isnan(x))[0]
    if idx.size < 2 * win + 1:
        return out
    xv = x[idx]
    med = ndimage.median_filter(xv, size=2 * win + 1, mode="nearest")
    mad = ndimage.median_filter(np.abs(xv - med), size=2 * win + 1, mode="nearest")
    sigma = 1.4826 * mad
    dev = np.abs(xv - med)
    out[idx] = (dev > nsigma * np.maximum(sigma, 1e-6)) & (dev > min_abs)
    return out


def _voiced_runs(voiced):
    runs, s = [], None
    for i, v in enumerate(voiced):
        if v and s is None:
            s = i
        elif not v and s is not None:
            runs.append((s, i)); s = None
    if s is not None:
        runs.append((s, len(voiced)))
    return runs


def _max_false_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if not v else 0
        best = max(best, cur)
    return best


def f0_spike_metrics(f0, cfg: Config = CFG):
    st = hz_to_semitone(f0)
    voiced = ~np.isnan(st)
    n = int(voiced.sum())
    if n < 5:
        return {"f0_spike_rate": np.nan, "f0_octave_jumps": np.nan, "n_voiced": n}
    spike = hampel_outliers(st, cfg.hampel_win, cfg.hampel_nsigma_f0, cfg.hampel_min_abs_f0)
    octv = 0
    for s, e in _voiced_runs(voiced):
        octv += int((np.abs(np.diff(st[s:e])) > cfg.octave_jump_semitone).sum())
    return {"f0_spike_rate": float(spike[voiced].sum()) / n, "f0_octave_jumps": octv, "n_voiced": n}


def energy_burst_metrics(rms_db, cfg: Config = CFG):
    burst = hampel_outliers(rms_db.astype(np.float64), cfg.hampel_win,
                            cfg.hampel_nsigma_energy, cfg.hampel_min_abs_energy)
    return {"energy_burst_rate": float(burst.mean())}


def vuv_and_pause_metrics(f0, rms_db, cfg: Config = CFG):
    voiced = f0 > 0
    flaps = int(np.sum(np.abs(np.diff(voiced.astype(int)))))
    dur = len(f0) * cfg.hop_ms / 1000.0
    flap_per_sec = flaps / max(dur, 1e-6)
    floor = rms_db.max() - cfg.silence_rel_db
    sp = np.where(rms_db > floor)[0]
    long_pause = 0.0
    if sp.size > 1:
        interior = (rms_db > floor)[sp[0]:sp[-1] + 1]
        long_pause = _max_false_run(interior) * cfg.hop_ms / 1000.0
    return {"vuv_flap_per_sec": flap_per_sec, "long_pause_sec": long_pause}


# ───────────────────────────── ref↔synth 분포 비교 ─────────────────────────────
def _f0_stats(f0):
    st = hz_to_semitone(f0)
    v = ~np.isnan(st)
    if v.sum() < 5:
        return np.nan, np.nan
    s = st[v]
    return float(np.median(s)), float(np.percentile(s, 75) - np.percentile(s, 25))


def prosody_match(ref_f0, syn_f0, ref_db, syn_db, cfg: Config = CFG):
    r_med, r_iqr = _f0_stats(ref_f0)
    s_med, s_iqr = _f0_stats(syn_f0)
    reg = abs(s_med - r_med) if not (np.isnan(s_med) or np.isnan(r_med)) else np.nan
    ratio = (s_iqr / r_iqr) if (r_iqr and not np.isnan(r_iqr) and r_iqr > 1e-3) else np.nan
    loud = abs(np.median(syn_db[syn_db > syn_db.max() - cfg.silence_rel_db]) -
               np.median(ref_db[ref_db > ref_db.max() - cfg.silence_rel_db]))
    return {"f0_register_delta_st": reg, "f0_range_ratio": ratio, "loudness_delta_db": float(loud)}


def reference_quality(ref, sr, cfg: Config = CFG):
    dur = len(ref) / sr
    clip = float(np.mean(np.abs(ref) >= cfg.clip_thr))
    db = frame_rms_db(ref, sr, cfg.hop_ms, cfg.win_ms)
    est_snr = float(np.percentile(db, 75) - np.percentile(db, 10))
    ok = (cfg.ref_min_dur <= dur <= cfg.ref_max_dur) and (clip <= cfg.max_clip_rate)
    return {"ref_dur": dur, "ref_clip_rate": clip, "ref_est_snr_db": est_snr,
            "ref_prefilter_ok": bool(ok)}


# ───────────────────────────── SECS (ECAPA + WavLM-SV) ─────────────────────────────
@lru_cache(maxsize=1)
def _ecapa(device="cpu"):
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                          run_opts={"device": device})


@lru_cache(maxsize=1)
def _wavlm_sv(device="cpu"):
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    m = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
    return fe, m, torch


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def secs_scores(ref, syn, sr, device="cpu"):
    out = {"secs_ecapa": np.nan, "secs_wavlm": np.nan}
    try:
        import torch
        enc = _ecapa(device)
        er = enc.encode_batch(torch.tensor(ref).unsqueeze(0)).squeeze().detach().cpu().numpy()
        es = enc.encode_batch(torch.tensor(syn).unsqueeze(0)).squeeze().detach().cpu().numpy()
        out["secs_ecapa"] = _cos(er, es)
    except Exception:
        pass
    try:
        fe, m, torch = _wavlm_sv(device)
        def emb(x):
            iv = {k: v.to(device) for k, v in fe(x, sampling_rate=sr,
                  return_tensors="pt", padding=True).items()}
            with torch.no_grad():
                return m(**iv).embeddings.squeeze().cpu().numpy()
        out["secs_wavlm"] = _cos(emb(ref), emb(syn))
    except Exception:
        pass
    return out


# ───────────────────────────── UTMOSv2 ─────────────────────────────
@lru_cache(maxsize=1)
def _utmos():
    import utmosv2
    return utmosv2.create_model(pretrained=True)


def utmos_score(synth_path):
    try:
        return float(_utmos().predict(input_path=synth_path))
    except Exception:
        return np.nan


# ───────────────────────────── ASR : CER + 국소 발화속도 ─────────────────────────────
_PUNCT = set("".join(chr(c) for c in range(0x21, 0x30)) + "".join(chr(c) for c in range(0x3A, 0x41)))


def normalize_ko(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "")
    return "".join(ch for ch in t if (not ch.isspace()) and (ch not in _PUNCT))


def to_jamo(text: str) -> str:
    try:
        from jamo import h2j
        return h2j(text)
    except Exception:
        return text


def _levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return np.nan
    try:
        import jiwer
        return float(jiwer.cer(ref, hyp))
    except Exception:
        return _levenshtein(ref, hyp) / len(ref)


@lru_cache(maxsize=1)
def _asr(device="cpu"):
    from faster_whisper import WhisperModel
    return WhisperModel("large-v3", device=device,
                        compute_type="float16" if device == "cuda" else "int8")


def asr_analyze(synth_path, device="cpu"):
    """returns (transcript, words[list of {text,start,end}]).  단어 타임스탬프 포함."""
    try:
        segs, _ = _asr(device).transcribe(synth_path, language="ko",
                                          beam_size=5, word_timestamps=True)
        transcript, words = "", []
        for s in segs:
            transcript += s.text
            for w in (s.words or []):
                words.append({"text": w.word, "start": w.start, "end": w.end})
        return transcript, words
    except Exception:
        return None, []


def flag_slow_words(words, cfg: Config = CFG):
    """국소 발화속도: 단어별 CPS 가 발화 중앙값 대비 과도하게 느린/끌린 단어 비율."""
    items = []
    for w in words:
        txt = normalize_ko(w["text"])
        dur = max(float(w["end"]) - float(w["start"]), 1e-6)
        nchar = max(len(txt), 1)
        items.append((nchar, dur, nchar / dur))
    if not items:
        return {"slow_word_rate": np.nan, "max_word_dur": np.nan, "n_words": 0}
    med = float(np.median([c for *_, c in items]))
    flagged, max_dur = 0, 0.0
    for nchar, dur, cps in items:
        slow = (cps < cfg.slow_word_frac * med) and (dur > cfg.slow_word_min_dur)
        drag = (dur > cfg.slow_word_abs_dur) and (nchar <= 2)
        flagged += int(slow or drag)
        max_dur = max(max_dur, dur)
    return {"slow_word_rate": flagged / len(items), "max_word_dur": max_dur, "n_words": len(items)}


def utterance_speaking_rate(n_samples, sr, target_text):
    return len(normalize_ko(target_text)) / max(n_samples / sr, 1e-6)


# ───────────────────────────── Stage 1 : 1:1 평가 ─────────────────────────────
def evaluate_stage1_pair(ref_path, synth_path, target_text, cfg=CFG, device="cpu"):
    ref = load_audio(ref_path, cfg.sr)
    syn = load_audio(synth_path, cfg.sr)
    syn_f0, _ = extract_f0(syn, cfg.sr, cfg, device)
    ref_f0, _ = extract_f0(ref, cfg.sr, cfg, device)
    syn_db = frame_rms_db(syn, cfg.sr, cfg.hop_ms, cfg.win_ms)
    ref_db = frame_rms_db(ref, cfg.sr, cfg.hop_ms, cfg.win_ms)
    row = {"ref_path": ref_path, "synth_path": synth_path}
    row.update(f0_spike_metrics(syn_f0, cfg))
    row.update(energy_burst_metrics(syn_db, cfg))
    row.update(vuv_and_pause_metrics(syn_f0, syn_db, cfg))
    row.update(prosody_match(ref_f0, syn_f0, ref_db, syn_db, cfg))
    row["speaking_rate_cps"] = utterance_speaking_rate(len(syn), cfg.sr, target_text)
    row.update(reference_quality(ref, cfg.sr, cfg))
    return row


def _dyn_rate_pass(series, cfg):
    import pandas as pd
    med = series.median()
    mad = (series - med).abs().median() * 1.4826
    if mad < 1e-9:
        return pd.Series(True, index=series.index)
    return ((series - med) / mad).abs() <= cfg.speaking_rate_mad_k


def run_stage1(manifest_csv, out_csv, cfg=CFG, device="cpu"):
    import pandas as pd
    mani = pd.read_csv(manifest_csv)
    rows = []
    for r in mani.itertuples(index=False):
        d = r._asdict()
        row = evaluate_stage1_pair(d["ref_path"], d["synth_path"], d["target_text"], cfg, device)
        row["dataset"] = d.get("dataset", "ALL")
        rows.append(row)
    df = pd.DataFrame(rows)

    grp = "dataset" if df["dataset"].nunique() > 1 else None
    if grp:
        df["rate_pass"] = df.groupby(grp, group_keys=False)["speaking_rate_cps"].apply(
            lambda s: _dyn_rate_pass(s, cfg))
    else:
        df["rate_pass"] = _dyn_rate_pass(df["speaking_rate_cps"], cfg)

    df["gate_f0"] = (df["f0_spike_rate"].fillna(1) <= cfg.max_f0_spike_rate) & \
                    (df["f0_octave_jumps"].fillna(99) <= cfg.max_f0_octave_jumps)
    df["gate_energy"] = df["energy_burst_rate"].fillna(1) <= cfg.max_energy_burst_rate
    df["gate_vuv"] = df["vuv_flap_per_sec"].fillna(99) <= cfg.max_vuv_flap_per_sec
    df["gate_pause"] = df["long_pause_sec"].fillna(99) <= cfg.max_long_pause_sec
    df["gate_ref"] = df["ref_prefilter_ok"].fillna(False)
    df["PASS"] = (df[["gate_f0", "gate_energy", "gate_vuv", "gate_pause", "gate_ref"]].all(axis=1)
                  & df["rate_pass"])

    df.to_csv(out_csv, index=False)
    passed = df.loc[df["PASS"], "ref_path"].tolist()
    pd.Series(passed, name="ref_path").to_csv(out_csv.replace(".csv", "_passed_refs.csv"), index=False)
    print(f"[Stage1] {len(df)} pairs | PASS {len(passed)} / {len(df)}")
    print(f"         pair scores  -> {out_csv}")
    print(f"         passed refs  -> {out_csv.replace('.csv', '_passed_refs.csv')}")
    return df, passed


# ───────────────────────────── Stage 2 : 1:다 평가 + 랭킹 ─────────────────────────────
def evaluate_stage2_pair(ref_path, synth_path, target_text, cfg=CFG, device="cpu"):
    ref = load_audio(ref_path, cfg.sr)
    syn = load_audio(synth_path, cfg.sr)
    syn_f0, _ = extract_f0(syn, cfg.sr, cfg, device)
    ref_f0, _ = extract_f0(ref, cfg.sr, cfg, device)
    syn_db = frame_rms_db(syn, cfg.sr, cfg.hop_ms, cfg.win_ms)
    ref_db = frame_rms_db(ref, cfg.sr, cfg.hop_ms, cfg.win_ms)

    row = {"ref_path": ref_path, "synth_path": synth_path}
    row.update(f0_spike_metrics(syn_f0, cfg))
    row.update(energy_burst_metrics(syn_db, cfg))
    row.update(vuv_and_pause_metrics(syn_f0, syn_db, cfg))
    row.update(prosody_match(ref_f0, syn_f0, ref_db, syn_db, cfg))
    row["speaking_rate_cps"] = utterance_speaking_rate(len(syn), cfg.sr, target_text)
    row.update(secs_scores(ref, syn, cfg.sr, device))
    row["secs"] = float(np.nanmean([row["secs_ecapa"], row["secs_wavlm"]]))
    row["utmos"] = utmos_score(synth_path)
    transcript, words = asr_analyze(synth_path, device)
    row["cer"] = cer(normalize_ko(target_text), normalize_ko(transcript))
    row["cer_jamo"] = cer(to_jamo(normalize_ko(target_text)), to_jamo(normalize_ko(transcript)))
    row.update(flag_slow_words(words, cfg))
    return row


def _pct_subscores(agg):
    """ref별 집계 raw → 항목별 백분위 점수(0~100, 높을수록 좋음)."""
    import pandas as pd
    sub = pd.DataFrame(index=agg.index)
    for item, members in ITEM_GROUPS.items():
        cols = []
        for col, orient in members:
            if col not in agg:
                continue
            s = agg[col].astype(float)
            s = -s if orient == "lower" else s
            s = s.fillna(s.min())
            cols.append(s.rank(pct=True) * 100.0)
        if cols:
            sub[item] = pd.concat(cols, axis=1).mean(axis=1)
    return sub


def run_stage2(manifest_csv, out_csv, cfg=CFG, device="cpu"):
    import pandas as pd
    mani = pd.read_csv(manifest_csv)
    rows = []
    for r in mani.itertuples(index=False):
        d = r._asdict()
        row = evaluate_stage2_pair(d["ref_path"], d["synth_path"], d["target_text"], cfg, device)
        row["dataset"] = d.get("dataset", "ALL")
        rows.append(row)
    pairs = pd.DataFrame(rows)
    pairs["cer_pass"] = pairs["cer"].fillna(1) <= cfg.max_cer
    pairs.to_csv(out_csv, index=False)

    raw_cols = ["secs", "utmos", "cer", "f0_spike_rate", "f0_octave_jumps",
                "energy_burst_rate", "vuv_flap_per_sec", "long_pause_sec",
                "slow_word_rate", "f0_register_delta_st", "loudness_delta_db"]
    agg = pairs.groupby("ref_path").agg(
        n=("synth_path", "size"),
        **{c: (c, "mean") for c in raw_cols if c in pairs}).reset_index().set_index("ref_path")

    sub = _pct_subscores(agg)
    w = {k: v for k, v in ITEM_WEIGHTS.items() if k in sub}
    wsum = sum(w.values())
    composite = sum((w[k] / wsum) * sub[k] for k in w)

    rank = sub.copy()
    rank.insert(0, "composite", composite)
    rank.insert(0, "n", agg["n"])
    for c in raw_cols:
        if c in agg:
            rank[f"raw_{c}"] = agg[c]
    rank = rank.sort_values("composite", ascending=False)
    rank.insert(0, "rank", range(1, len(rank) + 1))
    rank.reset_index().to_csv(out_csv.replace(".csv", "_ranking.csv"), index=False)

    print(f"[Stage2] {len(pairs)} pairs | {len(rank)} references ranked")
    print(f"         pair scores -> {out_csv}")
    print(f"         ranking     -> {out_csv.replace('.csv', '_ranking.csv')}")
    return pairs, rank.reset_index()


# ───────────────────────────── 랭킹 출력 ─────────────────────────────
def print_ranking(rank_df, start=1, end=10, per_metric=True):
    import pandas as pd
    if isinstance(rank_df, str):
        rank_df = pd.read_csv(rank_df)
    view = rank_df[(rank_df["rank"] >= start) & (rank_df["rank"] <= end)]
    items = [c for c in ITEM_GROUPS if c in rank_df.columns]
    print(f"\n=== Reference 랭킹  {start}~{end}위 ===")
    for _, r in view.iterrows():
        name = os.path.basename(str(r["ref_path"]))
        print(f"\n#{int(r['rank']):>3}  {name}   composite={r['composite']:.1f}  (n={int(r['n'])})")
        if per_metric:
            for it in items:
                lvl = int(round(r[it] / 10))
                bar = "#" * lvl + "." * (10 - lvl)
                print(f"        {it:<16} {bar} {r[it]:5.1f}")
    print()


# ───────────────────────────── CLI ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="2-stage TTS reference validator")
    ap.add_argument("--stage", type=int, choices=[1, 2])
    ap.add_argument("--manifest")
    ap.add_argument("--out", default="scores.csv")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--top", type=int, default=10, help="Stage2: 상위 몇 위까지 출력")
    ap.add_argument("--per-metric", action="store_true", help="항목별 점수 표시")
    ap.add_argument("--show", help="저장된 *_ranking.csv 를 불러와 순위만 조회")
    ap.add_argument("--rank-start", type=int, default=1)
    ap.add_argument("--rank-end", type=int, default=10)
    args = ap.parse_args()

    if args.show:
        print_ranking(args.show, args.rank_start, args.rank_end, True)
        return
    if args.stage == 1:
        run_stage1(args.manifest, args.out, CFG, args.device)
    elif args.stage == 2:
        _, rank = run_stage2(args.manifest, args.out, CFG, args.device)
        print_ranking(rank, 1, args.top, True)
    else:
        ap.error("--stage {1,2} 또는 --show 중 하나가 필요합니다.")


if __name__ == "__main__":
    main()
