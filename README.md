# Tandem

Developer Names:
Justin Kwinecki,
Zayed Umer,
William Pol,
Manav Patel and
Musab Mondal

Date of project start:
September 21, 2026

This project is ...

The folders and files for this project are as follows:

docs - Documentation for the project
refs - Reference material used for the project, including papers
src - Source code
test - Test cases
etc.

The documentation for this project is updated on the project's [GitHub page](https://smiths.github.io/capTemplate/). <!-- update for your project!  -->
## Architecture

The system integrates semantic routing with a KV-cache prediction
layer to investigate reducing redundant prefill computation when
switching between language models.

```mermaid
flowchart TD
    A["User sends prompt"] --> B["vLLM Semantic Router"]
    B --> C["Returns selected model"]
    C --> D{"Same as active model?"}

    D -->|Yes| E["Reuse valid KV cache"]
    D -->|No| F["Retrieve previous model's KV cache"]

    F --> G["Predict destination model's KV cache"]
    G --> H{"Prediction valid and useful?"}

    H -->|Yes| I["Load predicted cache into destination runtime"]
    H -->|No| J["Fallback: recompute prefill normally"]

    E --> K["Process remaining tokens"]
    I --> K
    J --> K

    K --> L["Destination model generates response"]

    style B fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    style G fill:#d1fae5,stroke:#059669,color:#064e3b
    style H fill:#fef3c7,stroke:#d97706,color:#78350f
    style J fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
```

> **Status:** Cross-model KV-cache prediction and reuse are proposed
> functionality and remain under investigation.