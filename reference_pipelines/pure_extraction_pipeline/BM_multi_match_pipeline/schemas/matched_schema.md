# Matched Facts Schema

```json
{
  "paper_id": "string",
  "target_id": "string",
  "canonical_name": "string",
  "matched_version": "v1",
  "accepted_candidates": [
    {
      "candidate_id": "string",
      "fact_type": "string",
      "match_decision": "accept",
      "match_confidence": 0.0,
      "reason": "string"
    }
  ],
  "rejected_candidates": [
    {
      "candidate_id": "string",
      "fact_type": "string",
      "match_decision": "reject",
      "match_confidence": 0.0,
      "reason": "string"
    }
  ],
  "ambiguous_candidates": [
    {
      "candidate_id": "string",
      "fact_type": "string",
      "match_decision": "ambiguous",
      "match_confidence": 0.0,
      "reason": "string"
    }
  ],
  "global_notes": ["string"]
}
```
