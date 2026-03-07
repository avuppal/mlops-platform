# MLOps Platform — Architecture

> All four diagrams render on GitHub natively via Mermaid.

---

## 1. Full MLOps Lifecycle

End-to-end flow from data ingestion through continuous monitoring and automated retraining.

```mermaid
flowchart TD
    A([Raw Data]) --> B[Feature Engineering]
    B --> C[Train Model\ntrain_fn]
    C --> D[Track Run\nExperimentTracker]
    D --> E[Evaluate\neval_fn]
    E --> F{Beats\nProduction?}
    F -- Yes --> G[Register Model\nModelRegistry]
    F -- No  --> H([Discard Candidate])
    G --> I[Promote → Staging]
    I --> J[Integration Tests]
    J --> K{Tests Pass?}
    K -- Yes --> L[Promote → Production]
    K -- No  --> M([Keep Current Production])
    L --> N[Serve Predictions]
    N --> O[Monitor\nDriftMonitor]
    O --> P{Drift or\nDegradation?}
    P -- No  --> O
    P -- Yes --> Q[Trigger Retraining\nRetrainingPipeline]
    Q --> C

    style A fill:#4CAF50,color:#fff
    style L fill:#2196F3,color:#fff
    style Q fill:#FF9800,color:#fff
    style H fill:#9E9E9E,color:#fff
    style M fill:#9E9E9E,color:#fff
```

---

## 2. Model Promotion Pipeline

Stage transitions within `ModelRegistry`, including the rollback escape hatch.

```mermaid
stateDiagram-v2
    [*] --> None : register_model()
    None --> Staging : promote("Staging")
    Staging --> Production : promote("Production")\n[auto-archives current Production]
    Production --> Archived : promote("Archived")\nor auto-archived on supersession
    Archived --> Production : rollback()\n[restores most-recent archive]

    note right of Production
        At most ONE version
        in Production at a time.
        rollback() is idempotent.
    end note
```

---

## 3. A/B Testing Decision Flow

How `ABTest` collects outcomes, runs Welch's t-test, and declares a winner.

```mermaid
flowchart LR
    subgraph Traffic Routing
        REQ([Incoming Request]) --> SPLIT{traffic_split\n50 / 50}
        SPLIT -- control  --> CM[Control Model\nmodel_v1]
        SPLIT -- treatment --> TM[Treatment Model\nmodel_v2]
    end

    subgraph Outcome Recording
        CM --> RC[record_outcome\ncontrol, value]
        TM --> RT[record_outcome\ntreatment, value]
        RC --> BUF[(Outcome Buffer)]
        RT --> BUF
    end

    subgraph Statistical Test
        BUF --> CHK{min_samples\nreached?}
        CHK -- No  --> BUF
        CHK -- Yes --> TTEST[Welch t-test\n_welch_t_test]
        TTEST --> PVAL[Compute p-value\nStudent-t CDF]
        PVAL --> SIG{p_value < α?}
        SIG -- No  --> INC([inconclusive])
        SIG -- Yes --> LIFT{treatment_mean\n> control_mean?}
        LIFT -- Yes --> WIN_T([Winner: treatment])
        LIFT -- No  --> WIN_C([Winner: control])
    end

    style WIN_T fill:#4CAF50,color:#fff
    style WIN_C fill:#2196F3,color:#fff
    style INC fill:#9E9E9E,color:#fff
```

---

## 4. Drift Detection Pipeline

How `DriftMonitor` detects distribution shift and triggers automated retraining.

```mermaid
flowchart TD
    subgraph Reference
        RD([Training Data\nReference Distribution])
        RD --> RF[Store per-feature\nreference arrays]
    end

    subgraph Serving Window
        SD([Serving Data\nCurrent Window])
        SD --> CF[Current feature\narrays]
    end

    RF --> PSI[PSI Calculation\nΣ Δ% × ln Δ%]
    CF --> PSI
    RF --> KS[KS Test\n2-sample ECDF distance]
    CF --> KS

    PSI --> PSI_THR{PSI ≥ 0.25?}
    KS  --> KS_THR{p-value < 0.05?}

    PSI_THR -- No  --> OK_PSI([No PSI Drift])
    PSI_THR -- Yes --> ALERT_PSI[⚠ PSI Drift Detected]
    KS_THR  -- No  --> OK_KS([No KS Drift])
    KS_THR  -- Yes --> ALERT_KS[⚠ KS Drift Detected]

    ALERT_PSI --> OR{overall_drifted?}
    ALERT_KS  --> OR
    OK_PSI --> OR
    OK_KS  --> OR

    OR -- True  --> TRIG([Trigger Retraining\nRetrainingPipeline.run])
    OR -- False --> WAIT([Continue Monitoring])

    TRIG --> PIPE[Train → Evaluate\n→ Register → Promote]

    style TRIG fill:#FF9800,color:#fff
    style WAIT fill:#4CAF50,color:#fff
    style ALERT_PSI fill:#f44336,color:#fff
    style ALERT_KS fill:#f44336,color:#fff
```
