# 06 Evaluation

最低限以下を別々に測定する。

## 圧縮

- dpc (decisions per character)
- chars / decision
- Chunk count
- average expanded length

## Generalization

- unseen pre-dpc
- compositional unseen
- rule unseen

## Identity

- resolve accuracy
- ambiguity
- substitution consistency

## Memory

- Recall success rate
- Recognition success rate
- hop count
- Fallback depth
- external lookup rate

## Route

- top-1 accuracy
- top-k accuracy
- route decision count

## 性能

- latency (ms/token)
- operations / input char
- bytes / Identity
- bytes / Relation
- HOT count / SLEEP count
- cache/locality benchmark
- snapshot size / load time
- train chars/sec
