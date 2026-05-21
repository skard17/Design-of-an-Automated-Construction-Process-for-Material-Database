# Step 10 Prompt Quality And Code Agent Flow

```mermaid
flowchart TD
    A["Step 9 output<br/>skyrmion_prompt_generation_output.json"] --> B["Load prompt package"]

    B --> C["Local format checks"]
    C --> C1["status == success"]
    C --> C2["validation_errors empty"]
    C --> C3["required modules present"]
    C --> C4["all schema fields covered<br/>104 / 104 currently"]
    C --> C5["execution_order valid"]
    C --> C6["aggregation exposes prompt_modules"]

    B --> D["Local quality lint"]
    D --> D1["JSON output contract"]
    D --> D2["evidence / provenance rules"]
    D --> D3["anti-hallucination rules"]
    D --> D4["field_path control"]
    D --> D5["section ownership control"]
    D --> D6["figure constraints"]
    D --> D7["repair prompt coverage"]
    D --> D8["length / empty / encoding checks"]

    C --> E["Base judgement"]
    D --> E

    E --> F{"--run-quality-review?"}
    F -- "no" --> G["Write judgement file<br/>prompt_quality_judgement.json"]

    F -- "yes" --> H["LLM quality review agent<br/>LiteLLM /chat/completions"]
    H --> H1["Model selection<br/>deepseek-v4 -> deepseek-v4-pro<br/>kimi-k2.6 currently works"]
    H1 --> I{"API succeeds?"}

    I -- "yes" --> J["Merge llm_quality_review"]
    J --> J1["overall_score"]
    J --> J2["dimension_scores"]
    J --> J3["quality_findings"]
    J --> J4["must_fix_before_code_generation"]
    J --> J5["code_generation_guidance"]
    J --> G

    I -- "no" --> K["Record llm_quality_review_error"]
    K --> K1["Current deepseek-v4-pro error:<br/>server 500, backend unreachable"]
    K --> G

    G --> L{"--judge-only?"}
    L -- "yes" --> M["Stop after judgement"]
    L -- "no" --> N["Code generation sub-agent"]
    N --> O["Build code-agent prompt from:<br/>prompt package + judgement + quality review"]
    O --> P["LiteLLM /chat/completions"]
    P --> Q["Write code_generation_agent_output.json"]
    Q --> R{"--write-generated-files?"}
    R -- "no" --> S["Keep generated code as JSON only"]
    R -- "yes" --> T["Write files under generated_code/<br/>does not overwrite original files"]
```

## Current Status

- Local format checks: passed.
- Local quality lint: passed.
- Field coverage: `104 / 104`.
- `deepseek-v4` maps to `deepseek-v4-pro`, but the API currently returns server `500` for that model.
- `kimi-k2.6` passed the minimal `/chat/completions` test and is currently the usable model.

## Main Commands

Local judgement only:

```powershell
python code\prompt_quality_code_agent.py `
  --prompt-output skyrmion_prompt_generation_output.json `
  --judgement-output prompt_quality_judgement.json `
  --judge-only
```

Judgement plus LLM quality review:

```powershell
$env:CODE_AGENT_API_KEY="your-key"
python code\prompt_quality_code_agent.py `
  --prompt-output skyrmion_prompt_generation_output.json `
  --judgement-output prompt_quality_judgement.json `
  --judge-only `
  --run-quality-review `
  --base-url https://chat.iphy.ac.cn/litellm/v1 `
  --model kimi-k2.6
```

Full judgement plus code-generation sub-agent:

```powershell
$env:CODE_AGENT_API_KEY="your-key"
python code\prompt_quality_code_agent.py `
  --prompt-output skyrmion_prompt_generation_output.json `
  --judgement-output prompt_quality_judgement.json `
  --code-agent-output code_generation_agent_output.json `
  --run-quality-review `
  --base-url https://chat.iphy.ac.cn/litellm/v1 `
  --model kimi-k2.6
```
