# Forensic Council — Technical Architecture Document
## Multi-Agent Forensic Evidence Analysis System · Version 3.0

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Foundational Architecture Principles](#2-foundational-architecture-principles)
3. [System Architecture Diagram](#3-system-architecture-diagram)
4. [Agent Definitions](#4-agent-definitions)
5. [Council Arbiter](#5-council-arbiter)
6. [Dual-Layer Memory Architecture](#6-dual-layer-memory-architecture)
7. [Human-in-the-Loop (HITL) Integration](#7-human-in-the-loop-hitl-integration)
8. [Inter-Agent Communication Protocol](#8-inter-agent-communication-protocol)
9. [Evidence Versioning & Chain of Custody](#9-evidence-versioning--chain-of-custody)
10. [Confidence Calibration Framework](#10-confidence-calibration-framework)
11. [Adversarial Robustness Architecture](#11-adversarial-robustness-architecture)
12. [Audit Trail Signing Architecture](#12-audit-trail-signing-architecture)
13. [XAI & Legal Admissibility Framework](#13-xai--legal-admissibility-framework)
14. [Technology Stack](#14-technology-stack)
15. [Risk Register](#15-risk-register)
16. [Phased Implementation Roadmap](#16-phased-implementation-roadmap)

---

## 1. System Overview

**Forensic Council** is a production-grade, court-admissible, AI-powered forensic evidence analysis platform. It deploys a **deliberative council of five specialist AI agents**, each independently investigating uploaded digital evidence through a structured ReAct reasoning loop. Their findings are synthesized by a **Council Arbiter** that moderates disagreements, executes challenge loops, and generates a single, cryptographically signed **XAI court-admissible forensic report**.

### Core Design Goals

| Goal | Description |
|---|---|
| **Court Admissibility** | Every finding traced through a logged, signed, stepwise reasoning chain |
| **Multi-Modal Coverage** | Image, audio, video, object, and metadata forensics in a single pipeline |
| **Deliberative Synthesis** | Contested findings challenged, re-examined, or escalated — never silently resolved |
| **Human Oversight** | Mandatory HITL checkpoints at defined severity and uncertainty thresholds |
| **Chain of Custody** | Every action, artifact, and intervention cryptographically signed and immutably stored |

---

## 2. Foundational Architecture Principles

These are non-negotiable cognitive and operational constraints that govern all design decisions.

| # | Principle | Core Constraint |
|---|---|---|
| 1 | **ReAct Loop Architecture** | Every agent operates exclusively on Thought → Action → Observation cycles. No single-pass classification. |
| 2 | **Dual-Layer Memory** | Working memory (within-session task state) + Episodic memory (cross-session case-linking). |
| 3 | **Human-in-the-Loop** | Hardcoded HITL checkpoints at severity thresholds, iteration ceilings, and unresolved contradictions. |
| 4 | **Task Decomposition** | Every agent decomposes its mandate into a structured sub-task list before entering its loop. |
| 5 | **Self-Reflection** | Mandatory structured self-critique pass before any agent submits findings to the Arbiter. |
| 6 | **Confidence Calibration** | Raw model scores are never court-admissible. All outputs pass through a calibration layer benchmarked against ground truth datasets. |
| 7 | **Evidence Versioning** | Every derivative artifact (crop, frame extract, audio segment) versioned and stored immutably. |
| 8 | **Adversarial Robustness** | Every agent checks findings against known anti-forensics evasion techniques before finalizing. |
| 9 | **Audit Trail Signing** | Every Thought, Action, Observation, tool call, and human intervention cryptographically signed at production time. |
| 10 | **Graceful Degradation** | Tool unavailability triggers explicit INCOMPLETE FINDING logging, not silent skipping. |
| 11 | **Agent Disagreement Tribunal** | Unresolvable inter-agent contradictions escalate to a human-chaired Tribunal. Human judgment is logged as an authoritative chain-of-custody event. |

---

## 3. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVIDENCE INGESTION LAYER                        │
│   Hash Computation · Immutable Storage · Version Root Creation          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   ORCHESTRATION LAYER    │
                    │  (LangGraph / AutoGen)   │
                    │  Async Multi-Agent Bus   │
                    └──┬──────┬──────┬──────┬─┘
                       │      │      │      │
         ┌─────────────┼──────┼──────┼──────┼─────────────┐
         │             │      │      │      │             │
    ┌────▼───┐   ┌─────▼─┐ ┌──▼──┐ ┌▼────┐ ┌▼────────┐  │
    │AGENT 1 │   │AGENT 2│ │AG 3 │ │AG 4 │ │AGENT 5  │  │
    │ Image  │   │ Audio │ │Obj/ │ │Video│ │Metadata │  │
    │Integrity│  │& Media│ │Wpn  │ │Temp │ │Context  │  │
    └────┬───┘   └───┬───┘ └──┬──┘ └──┬──┘ └────┬────┘  │
         │           │        │        │          │       │
    ┌────▼───────────▼────────▼────────▼──────────▼────┐ │
    │              ReAct Loop Engine (per agent)        │ │
    │  THOUGHT → ACTION → OBSERVATION → THOUGHT → ...  │ │
    │                                                   │ │
    │  ┌──────────────┐   ┌──────────────────────────┐ │ │
    │  │Working Memory│   │  Tool Execution Layer    │ │ │
    │  │(Task List)   │   │  + Graceful Degradation  │ │ │
    │  └──────────────┘   └──────────────────────────┘ │ │
    │  ┌──────────────────────────────────────────────┐ │ │
    │  │ Adversarial Robustness Check                 │ │ │
    │  └──────────────────────────────────────────────┘ │ │
    │  ┌──────────────────────────────────────────────┐ │ │
    │  │ Self-Reflection Pass                         │ │ │
    │  └──────────────────────────────────────────────┘ │ │
    │  ┌──────────────────────────────────────────────┐ │ │
    │  │ Confidence Calibration Layer                 │ │ │
    │  └──────────────────────────────────────────────┘ │ │
    └───────────────────────┬───────────────────────────┘ │
                            │                             │
         ┌──────────────────▼──────────────────────────┐  │
         │            HITL CHECKPOINT LAYER             │  │
         │  Investigator Briefing · Pause / Resume     │◄─┘
         │  Override · Redirect · Terminate · Escalate  │
         └──────────────────┬──────────────────────────┘
                            │
         ┌──────────────────▼──────────────────────────┐
         │              COUNCIL ARBITER                  │
         │  Cross-Agent Finding Comparison               │
         │  Challenge Loop (re-examination requests)     │
         │  Confidence Weighting & Cross-Modal Boost     │
         │  XAI Explanation Layer Generation             │
         └──────────────────┬──────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │  TRIBUNAL (if triggered)   │
              │  Human as Final Arbiter    │
              │  Contradiction Interface   │
              └─────────────┬─────────────┘
                            │
         ┌──────────────────▼──────────────────────────┐
         │         REPORT GENERATION LAYER              │
         │  Structured XAI Report Assembly              │
         │  Chain-of-Custody Compilation                │
         │  Evidence Version Tree Embedding             │
         │  Cryptographic Signing                       │
         └──────────────────┬──────────────────────────┘
                            │
         ┌──────────────────▼──────────────────────────┐
         │     FINALIZED COURT-ADMISSIBLE REPORT        │
         └─────────────────────────────────────────────┘
```

### Inter-Agent Communication Flows

```
Agent 2 (Audio)  ←──────────────────────────→  Agent 4 (Video)
  Cross-modal timestamp verification calls (bidirectional, mid-loop)

Agent 3 (Object) ──────────────────────────→  Agent 1 (Image)
  Lighting inconsistency → compositing check request (one-way)

Agent 4 (Video)  ──────────────────────────→  Agent 2 (Audio)
  Suspicious frame timestamp → audio splice cross-verification (one-way)

Arbiter          ──────────────────────────→  Any Specialist Agent
  Challenge loop: re-examination with additional context (one-way)
```

---

## 4. Agent Definitions

### Agent 1 — Image Integrity Agent
*Pixel-level forensic expert*

**Mandate:** Detect manipulation, splicing, compositing, and anti-forensics evasion at the pixel level.

**Task Decomposition (every session):**
```
[ ] Run full-image ELA and map anomaly regions
[ ] Isolate and re-analyze all flagged ROIs with noise footprint analysis
[ ] Run JPEG ghost detection on all flagged regions
[ ] Run frequency domain analysis on contested regions
[ ] Verify file hash against ingestion hash
[ ] Run adversarial robustness check against known anti-ELA evasion techniques
[ ] Self-reflection pass
[ ] Submit calibrated findings to Arbiter
```

**ReAct Loop Summary:**
Full-image ELA → ROI extraction → Noise footprint analysis → JPEG ghost detection → Frequency domain pass → Adversarial robustness check → Self-reflection → Calibration → Submit

**HITL Triggers:** Confirmed manipulation above severity threshold · Conflicting passes on same region · Tool unavailability on a mandatory sub-task

**Permitted Tool Actions:** Full-image forensic passes · ROI extraction and sub-region re-analysis · JPEG ghost detection · Frequency domain analysis · Noise fingerprint comparison · File hash verification · Camera sensor noise profile database query · Adversarial robustness check

**External Search:** Device sensor database · Forensic manipulation signature databases

**Inter-Agent:** Receives challenge requests from Agent 3 (lighting/compositing) and from the Arbiter

**Escalation Logic:** Conflicting passes on same region → CONTESTED FINDING with confidence range. Anti-forensics evasion suspected → ROBUSTNESS CAVEAT flagged.

---

### Agent 2 — Audio & Multimedia Forensics Agent
*Audio authenticity and multimedia consistency expert*

**Mandate:** Detect audio deepfakes, splices, re-encoding events, prosody anomalies, and audio-visual sync breaks.

**Task Decomposition (every session):**
```
[ ] Run speaker diarization — establish voice count baseline
[ ] Run anti-spoofing detection on primary speaker segments
[ ] Run prosody analysis across full track
[ ] Run background noise consistency analysis — identify shift points
[ ] Run codec fingerprinting for re-encoding event detection
[ ] Run audio-visual sync verification against video track timestamps
[ ] Issue collaborative call to Agent 4 for any flagged timestamps
[ ] Run adversarial robustness check against known anti-spoofing evasion
[ ] Self-reflection pass
[ ] Submit calibrated findings to Arbiter
```

**ReAct Loop Summary:**
Speaker diarization → Anti-spoofing per speaker → Prosody analysis → Background noise consistency → Codec fingerprinting → Audio-visual sync → Collaborative call to Agent 4 → Adversarial robustness check → Self-reflection → Calibration → Submit

**HITL Triggers:** Cross-modal confirmed splice above severity threshold · Inter-agent contradiction with Agent 4 · Tool unavailability affecting sync verification

**Permitted Tool Actions:** Full-track and segment-level anti-spoofing · Speaker diarization · Prosody analysis · Background noise consistency per time window · Codec fingerprinting · Audio-visual sync verification · Inter-agent call to Agent 4 · Adversarial robustness check

**External Search:** Voice synthesis artifact signature databases · Environmental audio libraries · Codec fingerprint databases

**Escalation Logic:** Anti-spoofing pass + prosody anomaly → both logged separately, never merged. Inter-agent contradiction with Agent 4 → INTER-AGENT CONTESTED FINDING to Arbiter.

---

### Agent 3 — Object & Weapon Analysis Agent
*Object identification and contextual validation specialist*

**Mandate:** Detect and contextually validate objects, weapons, and contraband. Identify compositing through lighting inconsistency.

**Task Decomposition (every session):**
```
[ ] Run full-scene primary object detection
[ ] For each detected object below confidence threshold: run secondary classification pass
[ ] For each confirmed object: run scale and proportion validation
[ ] For each confirmed object: run lighting and shadow consistency check
[ ] Run scene-level contextual incongruence analysis
[ ] Cross-reference confirmed objects against contraband/weapons database
[ ] Issue inter-agent call to Agent 1 for any region showing lighting inconsistency
[ ] Run adversarial robustness check against object detection evasion
[ ] Self-reflection pass
[ ] Submit calibrated findings to Arbiter
```

**ReAct Loop Summary:**
Full-scene detection → Secondary classification (if below threshold) → Scale validation → Lighting consistency → Scene incongruence → Database cross-reference → Collaborative call to Agent 1 (if compositing suspected) → Adversarial robustness check → Self-reflection → Calibration → Submit

**Conservative Threshold Principle:** Architecturally configured with a higher confidence acceptance threshold than all other agents. Every finding must pass: *"Is my confidence defensible in a court context?"*

**HITL Triggers:** Any confirmed weapon or contraband detection (mandatory) · Lighting inconsistency post-weapon confirmation · Secondary classification still below threshold after two passes

**Permitted Tool Actions:** Full-scene object detection · Isolated bounding box secondary classification · Scale and proportion validation · Lighting and shadow consistency check · Scene-level contextual incongruence analysis · Contraband and weapons database cross-reference · Inter-agent call to Agent 1 · Adversarial robustness check

**External Search:** Visual reference databases · Legal and jurisdictional databases · Forensic image databases for compositing signatures

**Escalation Logic:** Secondary classification below threshold → UNCONFIRMED — POSSIBLE DETECTION. Lighting inconsistency post-confirmation → COMPOSITING SUSPECTED with inter-agent call triggered.

---

### Agent 4 — Temporal Video Analysis Agent
*Temporal consistency and video integrity expert*

**Mandate:** Detect frame-level edit points, deepfake face swaps, optical flow anomalies, rolling shutter violations, and cross-modal temporal inconsistencies.

**Task Decomposition (every session):**
```
[ ] Run full-timeline optical flow analysis and generate temporal anomaly heatmap
[ ] For each flagged anomaly window: extract frames and run frame-to-frame consistency analysis
[ ] Classify each anomaly as EXPLAINABLE or SUSPICIOUS
[ ] For frames containing human faces: run face-swap detection
[ ] For each suspicious anomaly: issue collaborative call to Agent 2 for audio cross-verification
[ ] Validate rolling shutter behavior and compression patterns against claimed device metadata
[ ] Run adversarial robustness check against optical flow evasion
[ ] Self-reflection pass
[ ] Submit calibrated findings to Arbiter with dual anomaly classification list preserved
```

**ReAct Loop Summary:**
Full-timeline optical flow → Frame window extraction → Frame consistency analysis → EXPLAINABLE/SUSPICIOUS classification → Face-swap detection → Collaborative call to Agent 2 → Rolling shutter validation → Adversarial robustness check → Self-reflection → Calibration → Submit

**Running Classification Requirement:** Maintains two distinct lists throughout its loop: **EXPLAINABLE ANOMALIES** and **SUSPICIOUS ANOMALIES**. These lists are never collapsed into a single score. Both pass to the Arbiter.

**HITL Triggers:** Cross-modal confirmed finding above severity threshold · Anomaly unclassifiable after full analysis · Tool unavailability affecting face-swap detection on a flagged frame

**Permitted Tool Actions:** Full-timeline optical flow analysis · Temporal anomaly heatmap generation · Arbitrary frame window extraction and analysis · Frame-to-frame consistency analysis · Face-swap detection on specific frame windows · Facial landmark geometry analysis · Rolling shutter validation · Inter-agent call to Agent 2 · Adversarial robustness check

**External Search:** Deepfake generation model fingerprint databases · Camera model specification databases

**Escalation Logic:** Cross-modal confirmed finding → AUTO-ELEVATED to HIGH CONFIDENCE. Unclassifiable anomaly → AMBIGUOUS — ESCALATED with full reasoning chain.

---

### Agent 5 — Metadata & Context Analysis Agent
*Digital footprint and provenance analyst*

**Mandate:** Analyze EXIF metadata, GPS-timestamp consistency, file structure integrity, steganographic content, and detect provenance fabrication.

**Task Decomposition (every session):**
```
[ ] Extract all EXIF fields — explicitly log expected-but-absent fields
[ ] Cross-validate GPS coordinates against timestamp timezone
[ ] Run astronomical API check for GPS location and claimed date
[ ] Run reverse image search for prior online appearances
[ ] Run steganography scan
[ ] Run file structure forensic analysis
[ ] Verify file hash against ingestion hash
[ ] Query device fingerprint database against claimed device model
[ ] Synthesize cross-field consistency verdict
[ ] Run adversarial robustness check against metadata spoofing techniques
[ ] Self-reflection pass
[ ] Submit calibrated findings to Arbiter
```

**Absence as Signal Principle:** Every expected-but-absent EXIF field is a mandatory Thought trigger. Absence is never logged as neutral.

**HITL Triggers:** Timestamp fabrication confirmed · Multiple absent fields suggesting stripping · Cross-field contradiction between two or more metadata fields

**Permitted Tool Actions:** Full EXIF extraction with absent-field logging · GPS-timestamp cross-validation · Astronomical data API queries · Reverse image search · Timezone validation · Device fingerprint database lookup · Steganography scan · File structure forensic analysis · File hash verification

**External Search:** Reverse image search APIs · Astronomical and geolocation APIs · Device fingerprint databases · Geospatial validation databases

**Escalation Logic:** Any metadata field contradicting another or an external source → CROSS-FIELD CONTRADICTION preserved in full. Multiple absent fields → METADATA STRIPPING SUSPECTED. Prior case match in episodic memory → case-linking flag.

---

## 5. Council Arbiter

*Chief forensic report generator and deliberation moderator*

The Arbiter is not a specialist agent. It does not analyze raw evidence. It receives calibrated, self-reflected findings from all five agents and operates its own ReAct loop to synthesize, challenge, and finalize the forensic record.

**Task Decomposition:**
```
[ ] Receive and index all agent findings with confidence scores
[ ] Run cross-agent finding comparison — map agreements, independent findings, contradictions
[ ] For each contested finding: issue challenge query to relevant agent
[ ] For unresolved contestations: trigger Agent Disagreement Tribunal
[ ] Weight all findings by confidence, calibration, and cross-modal confirmation status
[ ] Generate XAI explanation layer for all findings
[ ] Compile full chain-of-custody log
[ ] Structure court-admissible report with all required sections
[ ] Append all agent ReAct reasoning chains as forensic appendices
[ ] Apply cryptographic signature to finalized report
```

**Challenge Loop:** The Arbiter can send any contested finding back to the originating agent with additional context from another agent's findings. The challenged agent re-enters a focused ReAct sub-loop. If the challenge resolves the contradiction, the Arbiter proceeds. If both agents maintain contradictory positions after re-examination, the Tribunal is triggered.

**Cross-Modal Confirmation Boost:** Findings independently confirmed by two or more agents examining different modalities (e.g., Agent 2 audio splice + Agent 4 visual edit point at same timestamp) are elevated to the highest confidence tier.

### Final Report Required Sections

| Section | Description |
|---|---|
| Executive Summary | Plain language findings with calibrated confidence statements |
| Per-Agent Findings | Full findings with calibrated confidence scores and robustness caveat status |
| Cross-Modal Confirmed Findings | Highlighted, highest confidence tier |
| Contested Findings | Both conflicting data points explicitly stated, never silently resolved |
| Tribunal-Resolved Findings | Attributed to human judgment, distinguished from agent-resolved findings |
| Incomplete Findings | All findings affected by tool unavailability |
| Case-Linking Flags | Prior case matches from episodic memory |
| Chain of Custody Log | Every action, inter-agent call, HITL checkpoint — all cryptographically signed |
| Evidence Version Tree | All derivative artifacts with parent-child relationships |
| Agent ReAct Reasoning Chains | Full Thought/Action/Observation logs per agent as forensic appendices |
| Self-Reflection Outputs | Per agent, as quality audit appendix |
| Uncertainty Statement | Formal statement of what the system was unable to determine and why |
| Cryptographic Signature Block | Report-level signature with verification instructions |

---

## 6. Dual-Layer Memory Architecture

### Working Memory (Within-Session)

**Purpose:** Tracks the active state of an agent's ReAct loop — sub-task decomposition list, in-progress and completed tasks, blocked tasks, and mid-loop observations.

**Structure:**
```
Working Memory Object (per agent, per session):
{
  session_id:        UUID
  agent_id:          agent identifier
  task_list: [
    { task_id, description, status: [PENDING|IN-PROGRESS|COMPLETE|BLOCKED], result_ref }
  ]
  current_loop_iteration:   integer
  iteration_ceiling:        integer
  active_roi_set:           [region references]
  pending_inter_agent_calls:[call references]
  hitl_checkpoint_state:    [null | PAUSED | RESUMED]
}
```

**Implementation Approach:** In-memory structured state (Redis or equivalent fast key-value store) with append-only log for forensic persistence. Every read and write signed as chain-of-custody event.

**Lifecycle:** Initialized at session start from task decomposition step. Cleared at session end. Serialized to persistent storage at every HITL checkpoint pause for resume integrity.

### Episodic Memory (Cross-Session)

**Purpose:** Enables case-linking across serial investigations. Agents query this store when they detect signatures encountered in prior cases.

**Index Keys:**
- Device fingerprint (EXIF camera model + noise profile hash)
- Metadata signature (absent-field pattern hash)
- Detected object class + scene composition hash
- Audio-visual artifact type (codec fingerprint, deepfake model signature)
- Manipulation artifact signature (ELA pattern, splice boundary profile)

**Implementation Approach:** Vector database (e.g., Weaviate or Qdrant) with forensic signature embeddings. Each memory entry stores: case ID, agent ID, session timestamp, finding type, confidence, and reference to the signed finding artifact.

**Chain of Custody:** Every episodic memory read and write logged and signed. Memory entries are immutable once written — updates create new versioned entries.

---

## 7. Human-in-the-Loop (HITL) Integration

### Mandatory HITL Trigger Conditions (All Agents)

| Trigger | Condition |
|---|---|
| **Iteration Ceiling** | Agent reaches 50% of iteration ceiling without a confirmed finding |
| **Contested Finding** | Agent's own loop cannot resolve a contradictory observation |
| **Tool Unavailability** | Required tool or API unavailable, materially impacting investigation |
| **Severity Threshold** | Confirmed weapon, timestamp fabrication, or other defined high-severity finding |
| **Unresolved Tribunal** | Arbiter challenge loop fails — escalates to human-chaired Tribunal |

### HITL Loop Pause & Resume Architecture

```
Agent ReAct Loop
       │
  HITL Trigger
       │
  ┌────▼────────────────────────────────┐
  │  PAUSE STATE                         │
  │  1. Serialize Working Memory to store│
  │  2. Log HITL checkpoint event        │
  │  3. Cryptographically sign log entry │
  │  4. Push Investigator Brief update   │
  │  5. Await human direction            │
  └────┬────────────────────────────────┘
       │
  Human Action
       │
  ┌────▼──────────────────────────────────────────────────────┐
  │ APPROVE & CONTINUE  │ REDIRECT  │ OVERRIDE  │ TERMINATE   │
  │  Restore Working    │  Inject   │  Log human│  Accept     │
  │  Memory from store  │  context  │  judgment │  current    │
  │  Resume loop        │  Resume   │  Resume   │  findings   │
  └────────────────────────────────────────────────────────────┘
```

### Investigator Briefing Interface

A continuously updated plain-language summary available at any time during active agent loops, not only at checkpoints. Contains: what the agent has found, what it is currently investigating, what it is uncertain about, and what it is blocked on. Human-readable — not a raw ReAct chain dump. Every briefing state is logged as part of the chain of custody.

### Agent Disagreement Tribunal Interface

Triggered when the Arbiter's challenge loop fails to resolve a contradiction.

Presents to the human investigator:
- The exact point of contradiction (timestamped, evidence-referenced)
- Agent A's full reasoning chain supporting its position
- Agent B's full reasoning chain supporting its contradictory position
- Confidence scores both agents assign to their positions
- The specific evidence artifacts each agent is relying on

Human determination is logged as an **authoritative human judgment** in the chain of custody, cryptographically signed, and distinguished from agent-resolved findings in the final report.

---

## 8. Inter-Agent Communication Protocol

### Permitted Communication Paths

| Caller | Callee | Trigger |
|---|---|---|
| Agent 2 | Agent 4 | Flagged audio timestamp → visual cross-verification request |
| Agent 4 | Agent 2 | Suspicious video frame timestamp → audio splice cross-verification |
| Agent 3 | Agent 1 | Lighting inconsistency detected → compositing/splicing check |
| Arbiter | Any Agent | Challenge loop — re-examination with additional context |

### Protocol Specification

**Call Structure:**
```
InterAgentCall {
  call_id:         UUID
  caller_agent_id: string
  callee_agent_id: string
  call_type:       [COLLABORATIVE | CHALLENGE]
  payload: {
    timestamp_ref:   (for temporal calls)
    region_ref:      (for spatial calls)
    context_finding: (for challenge calls)
    question:        plain-language investigation request
  }
  chain_of_custody_ref: signed log entry reference
}
```

**Anti-Circular Dependency Rules:**
- Agent 2 ↔ Agent 4 calls are bidirectional but not recursive. Agent 2 may call Agent 4 once per loop; Agent 4 may return findings but may not re-initiate a call to Agent 2 on the same artifact within that loop.
- Agent 3 → Agent 1 is one-way. Agent 1 does not call Agent 3.
- Arbiter challenge calls are terminal — a challenged agent cannot re-challenge.
- All inter-agent calls logged in the calling agent's ReAct chain and in the global chain-of-custody log.

---

## 9. Evidence Versioning & Chain of Custody

### Evidence Version Tree

Every evidence artifact (original + derivative) stored as a node in an immutable version tree.

```
Root Evidence Artifact
│   hash: SHA-256(original)
│   ingestion_timestamp
│   ingestion_agent_id
│
├── Derivative: ELA Output Image (Agent 1, Pass 1)
│   │   parent_ref: root
│   │   action: ELA_FULL_IMAGE
│   │   agent_id, timestamp, signed
│   │
│   └── Derivative: ROI Crop (Agent 1, Pass 2)
│           parent_ref: ELA output
│           action: ROI_EXTRACT
│           bounding_box coordinates
│           agent_id, timestamp, signed
│
├── Derivative: Audio Segment Extract (Agent 2, 00:38–00:46)
│       parent_ref: root
│       action: AUDIO_SEGMENT_EXTRACT
│       time_range: [00:38, 00:46]
│       agent_id, timestamp, signed
│
└── Derivative: Video Frame Window (Agent 4, frames 840–855)
        parent_ref: root
        action: FRAME_WINDOW_EXTRACT
        frame_range: [840, 855]
        agent_id, timestamp, signed
```

**Immutability Enforcement:** Append-only content-addressed storage (e.g., IPFS or equivalent). No artifact is deleted. Storage hash embedded in each version node.

### Chain of Custody Log Structure

Every loggable event in the system produces a signed log entry:

```
ChainOfCustodyEntry {
  entry_id:       UUID
  entry_type:     [THOUGHT | ACTION | OBSERVATION | TOOL_CALL | INTER_AGENT_CALL
                   HITL_CHECKPOINT | HUMAN_INTERVENTION | MEMORY_READ | MEMORY_WRITE
                   ARTIFACT_VERSION | CALIBRATION | SELF_REFLECTION | FINAL_FINDING
                   TRIBUNAL_JUDGMENT | REPORT_SIGNED]
  agent_id:       string
  timestamp:      ISO 8601 UTC
  content:        structured payload
  content_hash:   SHA-256(content)
  signature:      ECDSA(agent_private_key, content_hash + timestamp)
  prior_entry_ref: previous entry hash (chain link)
}
```

---

## 10. Confidence Calibration Framework

### Calibration Pipeline (Per Agent)

```
Raw Model Score → Calibration Model → Calibrated Probability → Court-Admissible Statement
```

**Calibrated Statement Format:**
> "Based on benchmark performance against [Dataset], a model confidence of [raw score] in this detection class corresponds to a true positive rate of [Y%] with a false positive rate of [Z%] at this threshold."

### Calibration Approaches by Agent

| Agent | Recommended Calibration Method | Rationale |
|---|---|---|
| Agent 1 (Image) | Platt Scaling / Isotonic Regression | Binary (manipulated / authentic) class distribution |
| Agent 2 (Audio) | Temperature Scaling | Multi-class softmax outputs from anti-spoofing models |
| Agent 3 (Object) | Isotonic Regression | Non-parametric, handles irregular calibration curves for rare classes |
| Agent 4 (Video) | Platt Scaling | Temporal anomaly scores with known threshold behavior |
| Agent 5 (Metadata) | Rule-based threshold calibration | Deterministic field-matching logic less amenable to probabilistic calibration |

### Calibration Versioning

Calibration models are versioned independently of detection models. If a calibration model is updated, prior findings retain references to the calibration version used at the time of production. Retroactive re-calibration of finalized findings is prohibited.

---

## 11. Adversarial Robustness Architecture

Each agent tests its own findings against known anti-forensics evasion techniques before finalizing. If a known evasion technique is plausible, this is flagged as a **ROBUSTNESS CAVEAT** in the finding — not a contradiction, but a disclosure the court must consider.

### Adversarial Robustness Catalog by Agent

| Agent | Known Evasion Techniques to Check |
|---|---|
| Agent 1 | Anti-ELA tools that normalize error levels post-manipulation · Consistent recompression to mask splice boundaries · Seam-blending texture synthesis |
| Agent 2 | Adversarial perturbations targeting anti-spoofing models · Noise injection to mask codec boundary signatures · Pitch/prosody normalization post-splice |
| Agent 3 | Adversarial patches applied to objects to fool classification · Object texture modification to reduce detection confidence · GAN-based texture synthesis for composited objects |
| Agent 4 | Optical flow smoothing applied at edit points to normalize anomaly signatures · Frame blending at splice boundaries · Temporal texture synthesis |
| Agent 5 | Metadata spoofing tools · Partial field stripping with plausible field retention · GPS coordinate injection with astronomical-aware timestamp alignment |

### Integration into ReAct Loop

Adversarial robustness check is a dedicated ACTION step in every agent's loop, executed after primary finding confirmation and before self-reflection. Its output is always logged — both detection of evasion (caveat flagged) and non-detection (finding confirmed as robust).

---

## 12. Audit Trail Signing Architecture

### Signing Architecture

**Algorithm:** ECDSA with P-256 curve (NIST-approved, court-familiar)

**Per-Entry Signing:** Every individual chain-of-custody log entry signed at the time of production using the producing agent's private key. Signing is synchronous and blocking — no entry is written without a valid signature.

**Chained Hash Structure:** Each entry embeds a hash of the prior entry, creating a tamper-evident chain. Any post-hoc modification to any entry breaks the chain at that point and all subsequent entries.

**Report-Level Signature:** The finalized court-admissible report receives an additional signature from the Arbiter, covering the full report content hash.

### Key Management

| Key Type | Management Approach |
|---|---|
| Agent Identity Keys | Generated at agent deployment. Stored in hardware security module (HSM) or equivalent. Rotated per case batch with prior key archived. |
| Arbiter Report Key | Separate key for report-level signatures. Higher custody requirements. |
| Human Intervention Keys | Investigator-specific key pair. Human interventions signed with investigator key, not agent key — distinguishing human from agent actions in the audit trail. |

### Tamper Evidence Surface

Signature verification produces a human-readable tamper report: which entry failed verification, what the content hash was at signing vs. at verification, and a timeline of the chain break. Presented in the final report's Chain of Custody section and available as a standalone verification artifact.

---

## 13. XAI & Legal Admissibility Framework

### Explainability Requirements

| Layer | XAI Approach |
|---|---|
| Per-Finding | Calibrated confidence statement with benchmark attribution |
| Per-Agent | Full Thought/Action/Observation chain as forensic appendix |
| Cross-Agent | Arbiter's reasoning for confidence weighting and cross-modal boosts |
| Contested | Explicit statement of both conflicting positions, not a merged score |
| Tribunal | Human judgment explicitly attributed and distinguished |

### Legal Admissibility Considerations

| Requirement | System Design Response |
|---|---|
| Reproducibility | Full reasoning chain logged and signed — any finding can be re-derived from its chain |
| Transparency | No black-box scores. Every classification linked to its reasoning steps |
| Human Oversight | HITL checkpoints at severity thresholds; human review of all contested findings |
| Chain of Custody | Tamper-evident signed log from evidence ingestion to report finalization |
| Expert Attribution | Agent identity, model version, calibration version all embedded in every finding |
| Uncertainty Disclosure | Mandatory Uncertainty Statement section; INCOMPLETE FINDINGs never suppressed |

---

## 14. Technology Stack

| Capability Layer | Recommended Technology | Justification |
|---|---|---|
| **Orchestration** | LangGraph | Native ReAct loop support, stateful graph execution, native HITL pause/resume, full reasoning chain persistence |
| **Agent Framework** | LangGraph agents over AutoGen | Deterministic loop control, explicit iteration ceiling enforcement, better forensic auditability |
| **Working Memory** | Redis (in-memory, with AOF persistence) | Sub-millisecond read/write mid-loop, append-only persistence for forensic logging |
| **Episodic Memory / Vector DB** | Qdrant or Weaviate | Forensic signature vector indexing, hybrid search (vector + metadata filter), self-hosted for air-gapped deployment |
| **Immutable Evidence Storage** | IPFS or S3 with object lock | Content-addressed immutability, version tree support |
| **Chain-of-Custody Logging** | Append-only PostgreSQL with pgcrypto | Relational structure for audit queries, native hash and signing support |
| **Cryptographic Signing** | Python `cryptography` library (ECDSA P-256) | NIST-standard, court-familiar, HSM-compatible |
| **Image Forensics (Agent 1)** | `errorlevelanalysis` / `forensically` / custom PyTorch ELA · `noise-print` for sensor noise | ROI-level invocation, open-source, calibration-compatible |
| **Audio Forensics (Agent 2)** | `SpeechBrain` (anti-spoofing, diarization) · `pyannote.audio` · `librosa` | Segment-level operation, SOTA anti-spoofing, open-source |
| **Object Detection (Agent 3)** | YOLOv8 / RT-DETR (primary) + CLIP (secondary classification) | Real-time bounding box, explainable secondary classifier, calibratable outputs |
| **Video Forensics (Agent 4)** | RAFT optical flow · `FaceForensics++`-trained deepfake detectors · `dlib` landmark | Frame-window invocation, SOTA deepfake detection |
| **Metadata Forensics (Agent 5)** | `ExifTool` · `exiftool-py` · `piexif` · Stego tools (`stegdetect`, `zsteg`) | Comprehensive EXIF coverage, absent-field detection, file structure analysis |
| **Reverse Image Search** | Google Vision API / SerpAPI (with local fallback) | Runtime querying, prior appearance dating |
| **Astronomical API** | USNO API / `astropy` | GPS-timestamp cross-validation |
| **Confidence Calibration** | `sklearn.calibration` (Platt, Isotonic) · `netcal` | Multi-class calibration, versioned calibration model support |
| **XAI / Explainability** | LIME · SHAP (image saliency) · Custom ReAct chain formatter | Pixel-level saliency for image findings, natural language chain formatting for report |
| **Report Generation** | Structured JSON → PDF pipeline (ReportLab / WeasyPrint) | Deterministic structured output, embeds chain-of-custody references |
| **GPU Infrastructure** | NVIDIA A100 / H100 (cloud) or RTX 4090 (air-gapped) | Multi-pass iterative inference per agent, concurrent multi-agent execution |

---

## 15. Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| **Model Hallucination in ReAct Reasoning** | Critical | Iteration ceiling enforcement · HITL checkpoints at 50% ceiling · Self-reflection mandatory · Calibration layer rejects uncalibrated scores |
| **Chain-of-Custody Tampering** | Critical | Per-entry cryptographic signing · Chained hash structure · HSM key management · Tamper evidence surfaces in report |
| **Tool Unavailability at Critical Step** | High | Graceful degradation protocol · INCOMPLETE FINDING logging · Tool availability pre-check before loop entry · Investigator briefing on any unavailability |
| **Circular Inter-Agent Dependency** | High | Explicit protocol constraints (Section 8) · Anti-circular rules enforced at orchestration layer · Arbiter challenge calls are terminal |
| **Calibration Drift Retroactively Affecting Prior Findings** | High | Calibration model versioning · Findings embed calibration version at production time · Retroactive re-calibration prohibited |
| **Working Memory Corruption on HITL Resume** | Medium | Serialized memory persistence at every checkpoint · Resume integrity check before loop continuation · Resume event signed as chain-of-custody entry |
| **Adversarial Evidence Defeating All Agents** | Medium | Per-agent adversarial robustness checks · ROBUSTNESS CAVEAT disclosure in report · Cross-modal confirmation requirement for highest confidence tier |
| **Episodic Memory Poisoning via False Prior Cases** | Medium | Episodic memory entries are immutable and signed · Memory writes require same authentication as primary findings · Query results surfaced to human investigator for review |
| **Key Compromise Undermining Prior Case Audits** | High | Per-case-batch key rotation · Prior keys archived in HSM · Signed verification artifacts stored separately from case files |
| **Tribunal Deadlock (Human Unavailable)** | Low | Tribunal escalation threshold is configurable · System can finalize report with Tribunal-Unresolved section if human review is deferred · Contested findings never suppressed |
| **GPU Latency Under Multi-Agent Concurrent Load** | Medium | Agent parallelism with async orchestration · GPU scheduling per agent mandate · Staged deployment (sequential prototype → concurrent production) |
| **Legal Admissibility Challenge to AI-Generated Report** | High | Full ReAct chain as forensic appendix · Calibrated confidence expressions with benchmark attribution · Human judgment clearly distinguished from agent findings · Tribunal mechanism demonstrates adversarial review |

---

## 16. Phased Implementation Roadmap

### Phase 0 — Infrastructure & Tooling Foundation (Weeks 1–4)
- Establish immutable evidence storage and version tree schema
- Implement chain-of-custody logging with cryptographic signing (per-entry ECDSA)
- Set up working memory (Redis) and episodic memory (Qdrant) infrastructure
- Instrument orchestration layer (LangGraph) with iteration ceiling and HITL pause/resume
- Define and validate inter-agent communication protocol

### Phase 1 — Single Agent Prototype (Weeks 5–10)
- Implement Agent 5 (Metadata) as the first full ReAct loop agent — deterministic logic makes it the lowest-risk validation case
- Validate: task decomposition → ReAct loop → working memory integration → HITL checkpoint → self-reflection → calibration → Arbiter submission
- Validate full chain-of-custody log production for a single agent session
- Validate evidence versioning for one derivative artifact type

### Phase 2 — Full Agent Council (Weeks 11–20)
- Implement Agents 1, 2, 3, 4 with full ReAct loops, tool integrations, and memory
- Implement adversarial robustness checks per agent
- Implement inter-agent communication (Agent 2 ↔ Agent 4, Agent 3 → Agent 1)
- Implement episodic memory querying and case-linking
- Validate HITL checkpoint and Investigator Briefing interface

### Phase 3 — Arbiter & Tribunal (Weeks 21–26)
- Implement Council Arbiter with cross-agent finding comparison and challenge loop
- Implement Agent Disagreement Tribunal interface
- Implement confidence weighting and cross-modal confirmation boosting
- Implement XAI explanation layer generation

### Phase 4 — Report Generation & Legal Hardening (Weeks 27–32)
- Implement structured court-admissible report generation with all required sections
- Implement report-level cryptographic signing
- Implement tamper evidence verification workflow
- Calibration model benchmarking per agent domain — ground truth dataset identification and calibration fitting
- Legal admissibility review with forensic domain expert

### Phase 5 — Production Hardening & Deployment (Weeks 33–40)
- Performance optimization for concurrent multi-agent execution
- Air-gapped deployment packaging (for law enforcement use cases)
- Security audit of key management and signing architecture
- Load testing with realistic evidence file sizes and multi-session episodic memory volumes
- Documentation of full system for court expert witness preparation

---

*Document Version: 1.0 · Architecture Phase · No code generated.*
*All design decisions governed by the 11 Foundational Architecture Principles.*
