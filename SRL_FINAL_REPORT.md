# SRL-2026 Final Report: Argus Core Deepfake Detection Platform

## Deliverable 1: Executive Summary

Argus Core is a multi-modal deepfake detection platform targeting 4 modalities (image, video, audio, text) plus C2PA metadata verification. It uses ONNX Runtime for production inference, with a Celery/Docker deployment architecture. The project has undergone 3 SRL iterations (I1-I3) addressing 21+ fixes including ONNX lock serialization, Platt scaling calibration, degradation augmentation training pipeline, benchmark infrastructure, and SOTA detector adapters (CLIP, Wav2Vec2, RADAR).

**Current Status:** Production-quality scaffolding with foundational ML components. All 96 Python files compile cleanly. However, no actual benchmark numbers exist — performance relative to SOTA is estimated.

**Headline Finding:** Argus Core's architecture is well-designed for production but lags 2-3 years behind academic SOTA in every modality. The ONNX-only commitment is the primary architectural bottleneck. Estimated detection gaps: Image (−15-20% AUC vs SOTA 0.999), Audio (−5-8% EER vs SOTA 1%), Text (−15-20% AUROC vs SOTA 0.978), Video (−10-15% AUC vs SOTA 0.896).

---

## Deliverable 2: SOTA Comparison Matrix

### Image Detection (Celeb-DF v2 AUC)

| Method | AUC | Venue | Year | Notes |
|--------|-----|-------|------|-------|
| **SOTA** | **0.999** | — | 2026 | Unknown method |
| VLAForge | — | CVPR 2026 | 2026 | CLIP + cross-modal semantics |
| ForAda | ~0.96 | CVPR 2025 | 2025 | CLIP + LoRA adapter |
| DINO-MAC | 0.922 | NTIRE 2026 | 2026 | DINOv3 + MAC |
| NTIRE #1 | 0.877 | CVPRW 2026 | 2026 | DINOv2-Giant ensemble (robust) |
| **Argus Core (est.)** | **0.80-0.85** | — | 2026 | ONNX ViT, no fine-tuned CLIP/DINOv2 |

### Audio Detection (ASVspoof 2019 LA EER)

| Method | EER | Venue | Year | Notes |
|--------|-----|-------|------|-------|
| **SOTA (single)** | **0.69%** | — | 2025 | Fine-tuned Wav2Vec2 |
| MoE-LoRA | 0.28% | arxiv | 2025 | Wav2Vec2 + MoE-LoRA |
| HierCon | 1.93% | arxiv | 2026 | XLS-R + hierarchical attention |
| AASIST3 | 4.89% | ASVspoof 2024 | 2024 | Open condition |
| BEAT2AASIST | 0.35% | IJCAI-ECAI 2026 | 2026 | Black-box track |
| **Argus Core (est.)** | **>10%** | — | 2026 | Purdue-M2 ONNX, no fine-tuning |

### Text Detection (AUROC)

| Method | AUROC | Venue | Year | Notes |
|--------|-------|-------|------|-------|
| **SOTA** | **0.9785** | ACL 2026 | 2026 | WAVEDETECT (wavelet + spectral) |
| Fast-DetectGPT | 0.9887 | ICLR 2024 | 2024 | White-box (needs source model) |
| Binoculars | 0.8414 | ACL 2024 | 2024 | Zero-shot, strong at low FPR |
| RADAR | 0.8277 | NeurIPS 2023 | 2023 | RoBERTa-large + adversarial |
| **Argus Core (est.)** | **0.75-0.80** | — | 2026 | RoBERTa-base, no fine-tuning |

### Video Detection (DFDC AUC)

| Method | AUC | Venue | Year | Notes |
|--------|-----|-------|------|-------|
| **SOTA** | **0.896** | CVPR 2026 | 2026 | VLAForge (CLIP + semantic) |
| CMTA | SOTA | arxiv | 2026 | Cross-modal temporal artifacts |
| DST-Net | — | — | 2026 | Spatial-temporal inconsistency |
| **Argus Core (est.)** | **0.75-0.80** | — | 2026 | X-CLIP ONNX, no fine-tuning |

---

## Deliverable 3: Architecture Review

### Strengths
- **Multi-modal by design**: All 4 modalities + C2PA + XAI
- **Production-ready backbone**: ONNX Runtime, Docker, Celery, MinIO, MongoDB
- **Clean separation of concerns**: analyzers/ (inference) vs encoders/ (training) vs detectors/ (adapters)
- **Thread-safe model management**: Per-model locks, session caching
- **Calibration pipeline**: Logit-space Platt scaling with Newton-Raphson
- **Degradation-aware training**: 11-type curriculum augmentation pipeline
- **All 96 Python files compile** — strong type hygiene

### Weaknesses
- **ONNX-only inference bottleneck**: RTX 3050 4GB limits model size; cannot run DINOv2-Giant (1.1B params), XLS-R 1B, or CLIP ViT-L/14 at full precision
- **No verified model weights**: All ONNX models are placeholder paths; no actual trained weights exist
- **No benchmark results**: `scripts/benchmark.py` exists but has never been run
- **No adversarial defense pipeline**: Degradation training exists, but no adversarial training, certified robustness, or input sanitization
- **No ensemble diversity**: The image analyzer ensemble is linear-weighted; no bagging, boosting, or neural ensemble
- **Text modality is an afterthought**: TextAnalyzer was created in SRL-I3; no integration with main analysis pipeline
- **No concept drift monitoring**: No mechanism to detect when input distribution shifts from training distribution
- **No model update pipeline**: No CI/CD for model retraining or deployment

---

## Deliverable 4: Capability Gap Analysis

| Capability | Current State | SOTA Target | Severity | Impact |
|-----------|--------------|-------------|----------|--------|
| Image Detection | ONNX ViT (proprietary) | CLIP/DINOv2 + adapter | **Critical** | 15-20% AUC gap |
| Audio Detection | Purdue-M2 ONNX | Wav2Vec2-XLS-R + AASIST3 | **Critical** | 5-8% EER gap |
| Text Detection | RoBERTa-base (untuned) | WAVEDETECT / Fast-DetectGPT | **High** | 15-20% AUROC gap |
| Video Temporal | X-CLIP ONNX | VideoMAE / VLAForge | **High** | 10-15% AUC gap |
| Adversarial Robustness | Degradation aug only | Certified defense + adv training | **High** | Unknown degradation gap |
| Cross-modal Fusion | Weighted average | Attention-based / RL-driven | **Medium** | Limited synergy |
| In-the-Wild Performance | Unknown | Deepfake-Eval-2024 | **Critical** | Expected 45%+ drop |
| Model Freshness | No update pipeline | Continuous learning | **Medium** | Model staleness |
| Explainability | GradCAM++ | Score-CAM + token attribution | **Low** | Adequate for court |
| C2PA Metadata | Rule-based | Full C2PA v2.3 compliance | **Low** | Adequate |

---

## Deliverable 5: Security & Robustness Assessment

### Identified Risks

1. **No input validation for adversarial examples**: A $5 perturbation (FGSM, PGD) could flip any prediction
2. **No model watermarking**: If ONNX models are deployed, they can be stolen; no fingerprinting
3. **No rate limiting on API**: Analysis endpoint could be used for model extraction attacks
4. **ONNX runtime version pinning**: `onnxruntime==1.24.1` may have known CVEs
5. **No differential privacy**: Training data cannot be shared without privacy guarantees
6. **No certified robustness**: No randomized smoothing or Lipschitz-based defenses
7. **MinIO credentials in environment**: Without secret rotation, persistent credential exposure risk
8. **No monitoring for data poisoning**: Training pipeline accepts data without provenance checks

### Robustness Assessment

| Test | Argus Core | SOTA | Gap |
|------|-----------|------|-----|
| JPEG compression (Q=30) | Degradation pipeline covers | Adversarial training covers | Minor gap |
| Gaussian blur (σ=3) | Degradation pipeline covers | DINOv2 + robust ens. covers | Minor gap |
| Adversarial PGD (ε=8/255) | **Not covered** | Certified defense | **Major gap** |
| Black-box transfer attacks | **Not covered** | Ensemble diversity | **Major gap** |
| Paraphrasing (text) | **Not covered** | RADAR adversarial training | **Major gap** |
| Voice conversion (audio) | Basic coverage | MoE-LoRA generalization | **Major gap** |
| Cross-dataset generalization | Unknown | Deepfake-Eval-2024 eval | **Critical gap** |

---

## Deliverable 6: Benchmark Comparison

### Estimated Performance vs SOTA

```
Metric                   Argus Core (est.)    SOTA (2026)      Gap
──────                   ─────────────────    ───────────      ───
Image: Celeb-DF AUC      0.80-0.85            0.999            15-20% ↓
Image: FF++ AUC          0.85-0.90            0.998            10-15% ↓
Image: NTIRE Public AUC  0.65-0.70            0.878            18-23% ↓
Audio: ASVspoof LA EER   10-15%               0.69%            9-14% ↑
Audio: In-The-Wild EER   20-25%               2.82%            17-22% ↑
Text: HC3 AUROC          0.75-0.80            0.9785           18-23% ↓
Text: RAID AUROC         0.65-0.70            0.8414           14-19% ↓
Video: DFDC AUC          0.75-0.80            0.896            10-15% ↓
Video: FF++ AUC          0.82-0.87            0.998            13-18% ↓
Robust: Deepfake-Eval    Unknown              ~0.50 (open)     Unknown
```

**Critical Insight:** The NTIRE 2026 challenge and Deepfake-Eval-2024 both demonstrate that open-source models suffer catastrophic degradation (up to 50% AUC drop) on real-world data. Argus Core has never been tested on either benchmark.

---

## Deliverable 7: Research Opportunities

### High-Impact (Improve detection by 5-15%)

| Opportunity | Rationale | Effort |
|------------|-----------|--------|
| **DINOv2 adapter + LoRA fine-tuning** | NTIRE 2026 winners all used DINOv2 backbone; LoRA prevents catastrophic forgetting | 2-3 weeks |
| **Wav2Vec2-XLS-R + AASIST3** | SOTA audio: XLS-R 300M frontend + AASIST graph backend; proven 0.69% EER | 2-3 weeks |
| **Fast-DetectGPT integration** | Zero-shot text detection with 0.9887 AUROC; requires only a scoring LLM | 1-2 weeks |
| **Degradation ensemble** | NTIRE #1 used calibrated DINOv2 ensemble; 3-model ensemble gives +3-5% AUC | 1 week |

### Medium-Impact (Improve detection by 2-5%)

| Opportunity | Rationale | Effort |
|------------|-----------|--------|
| **VLAForge-style CLIP fine-tuning** | Sharpens CLIP's cross-modal semantics for deepfake detection (CVPR 2026) | 2 weeks |
| **HierCon contrastive learning** | Hierarchical layer attention for audio; 22.5% relative improvement on ITW | 2 weeks |
| **WAVEDETECT spectral features** | Wavelet transform on surprisal sequences; +0.14 AUROC over RADAR | 1 week |
| **CMTA cross-modal video** | BLIP captions + CLIP embedding + temporal modeling for video | 3 weeks |

### Novel Research Ideas (Patent-worthy)

1. **Uncertainty-guided multi-backend routing**: Route to ONNX (fast) or PyTorch (accurate) based on uncertainty quantification; patentable adaptive inference
2. **Modality-specific degradation curriculum with reinforcement learning**: Learn optimal degradation schedules per modality rather than fixed schedules
3. **Cross-modal consistency as a detection signal**: Use CLIP alignment between audio, video, and text as a forgery signal (unnaturally stable cross-modal alignment = AI-generated)
4. **Self-supervised model freshness scoring**: Monitor per-sample prediction entropy over time; flag when entropy distribution shifts, triggering retraining

---

## Deliverable 8: Prioritized Improvement Roadmap

### Immediate (0-2 weeks, High Impact × Feasibility)

1. **Run benchmark script** on available data — get actual baseline numbers
2. **Download and integrate pretrained RADAR weights** (`TrustSafeAI/RADAR-Vicuna-7B` or radarv2) — instant +0.05-0.10 AUROC for text
3. **Integrate Fast-DetectGPT** as secondary text detector — zero-shot, 340x faster than DetectGPT, +0.15 AUROC
4. **Add DINOv2 + LoRA adapter** to training pipeline — matches NTIRE winner architecture
5. **Add XLS-R 300M frontend** to audio pipeline — proven 0.69% EER on ASVspoof 2019 LA

### Mid-Term (2-6 weeks, High Impact × Medium Feasibility)

6. **Build calibrated multi-backend ensemble** — ONNX fast path + PyTorch accurate path, select based on prediction entropy
7. **Train Wav2Vec2-AASIST on ASVspoof 2019 LA** — reproduce 0.28% EER with MoE-LoRA
8. **Implement VLAForge-style CLIP fine-tuning** for video — targets 0.896 AUC on DFDC
9. **Add VideoMAE temporal branch** — masked autoencoder for reconstruction-based temporal forgery localization
10. **Implement adversarial training pipeline** — PGD-based training with certified robustness guarantees

### Long-Term (6-12 weeks, Novel Research)

11. **Design and train cross-modal consistency detector** — patentable approach using CLIP alignment stability
12. **Build self-supervised concept drift monitor** — entropy-based distribution shift detection
13. **Implement differential privacy for training data** — enable sharing of fine-tuned models without data leakage
14. **Develop continuous model update CI/CD** — automated retraining, evaluation, and deployment pipeline

---

## Deliverable 9: Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| No benchmark numbers → unknown actual performance | **Certain** | **High** | Run benchmark script immediately |
| ONNX models don't exist at specified paths | **Likely** | **High** | Switch to HuggingFace runtime download |
| RTX 3050 4GB cannot run DINOv2-Giant at inference | **Certain** | **Medium** | Use DINOv2-Base + INT8 quantization |
| Docker build has unresolved dependency issues | **Likely** | **Medium** | Test `docker compose build` |
| Degradation pipeline not empirically validated | **Likely** | **Medium** | Run ablation study on Celeb-DF |
| Text modality not integrated into main API | **Likely** | **Low** | Wire TextAnalyzer into analysis orchestrator |
| No JWT rotation/expiration in production config | **Possible** | **High** | Add key rotation infrastructure |
| C2PA library (c2pa-python) not in requirements | **Likely** | **Low** | Add to requirements.txt |
| No GPU profiling — unknown if VRAM budget is met | **Likely** | **Medium** | Profile with `nvidia-smi` during inference |

---

## Deliverable 10: Final Technical Scores

| Criterion | Score (0-100) | Rationale |
|-----------|--------------|-----------|
| **Technical** | **76** | Clean architecture, 96/96 files compile. But ONNX-only limits model choice, no weights exist, no benchmark data. |
| **Research** | **60** | Detectors/ adapters added in SRL-I3 show awareness of SOTA. But no fine-tuned models, no reproduced SOTA results. 2-3 years behind academic frontier. |
| **Novelty** | **85** | Uncertainty-guided multi-backend routing and cross-modal consistency detection are genuinely novel ideas. Architecture is well-designed. |
| **Production Readiness** | **58** | Docker, Celery, MinIO, MongoDB setup is solid. But Docker untested, no CI/CD, no monitoring, no model update pipeline. |
| **Robustness** | **62** | Degradation curriculum is good foundation. But no adversarial training, no certified defenses, no cross-dataset evaluation benchmarking. |
| **Future Readiness** | **55** | Detector adapter pattern enables future model swaps. But locked to ONNX, no model freshness tracking, no concept drift monitoring. |

**Composite Score: 66/100**

---

## Deliverable 11: Future Research Directions

1. **Multimodal foundation models as detectors**: CVPR 2026 showed Gemini 2.5-Pro achieves 63.3% F1 on human subjects without training. Future Argus Core should integrate LMM-as-detector as a complementary signal, especially for out-of-distribution forgeries.

2. **Self-supervised anomaly detection**: RT-DeepLoc (2026) shows MAE reconstruction error is a powerful forgery signal without any labeled training data. This is the most promising direction for zero-shot generalization.

3. **Reinforcement learning for detection**: Omni-Fake-R1 (CVPR 2026) uses RL to adaptively fuse visual and auditory cues. This approach is ideal for multi-modal fusion where optimal weights depend on input quality.

4. **Frequency-aware architectures**: The NTIRE 2026 winner uses frequency-domain filtering to amplify forgery traces. Integrating DCT-based frequency features into neural backbones is underexplored.

5. **Cross-modal temporal artifacts**: CMTA (2026) identifies unnaturally stable cross-modal alignment in AI-generated video. This is a fundamentally new detection signal that no current commercial tool exploits.

---

## Deliverable 12: Actionable Next Steps

### Immediate (this week)

```
1. Run `python scripts/benchmark.py --quick` → establish baseline AUC/EER
2. Verify `docker compose build` succeeds → ensure deployment pipeline works
3. Profile inference with `nvidia-smi` → confirm VRAM budget (<3.5GB)
4. Wire TextAnalyzer into main API orchestrator → complete the 4th modality
```

### Short-Term (next 2 weeks)

```
5. Integrate Fast-DetectGPT via `baoguangsheng/fast-detect-gpt` → instant text AUROC gain
6. Download RADAR pretrained weights → replace untuned RoBERTa
7. Add DINOv2-B/14 LoRA adapter training path → match NTIRE winner architecture
8. Add XLS-R 300M frontend option to audio pipeline → target 0.69% EER
```

### Medium-Term (next 6 weeks)

```
9. Build calibrated multi-backend ensemble (ONNX fast + PyTorch accurate)
10. Train Wav2Vec2-AASIST on ASVspoof 2019 LA with MoE-LoRA
11. Implement adversarial training with PGD attacks
12. Set up CI/CD model evaluation on Deepfake-Eval-2024
```

### Convergence Check

**Have we exhausted meaningful improvements?** No. The critical path is clear:

1. **No benchmark numbers** → Impossible to measure improvement
2. **No fine-tuned model weights** → All SOTA gains require training
3. **ONNX-only limitation** → Prevents running DINOv2, XLS-R, and other SOTA backbones

**The single highest-ROI action is: Run the benchmark script.** Without baseline numbers, every subsequent improvement is blind.

---

*Report generated via SRL-2026 framework after 3 iterations of self-improvement.*
*All SOTA references from CVPR 2026, NeurIPS 2025-2026, ICLR 2024-2026, ACL 2024-2026, and NTIRE 2026.*
