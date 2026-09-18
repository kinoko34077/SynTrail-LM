# 03 Learning & Stabilization

## 19. Merge / Factorization

**状態: 原理確定・新方式未実装**

MergeだけをA+B→ABと一方向に進めない。

```
Aは / Bは / Cは → A+は / B+は / C+は
```

を競争させる。

**概念的Gain:**
```
Gain = ProcessingSaving + ReuseGain + ConsistencyGain
     - StorageCost - RedundancyCost - ContradictionCost
```

係数は調整対象。

---

## 20. Context Diversity

**状態: 確定・未実装**

Contentについて以下を評価可能とする:

- use_count
- left_context_diversity
- right_context_diversity
- cross-context reuse

頻度だけでChunk価値を決めない。

---

## 21. Experience / Replay

**状態: 確定方針・実装中 (Phase 1)**

**Experience**: 外界から新しく得た観測。

更新対象:
- 世界の出現統計
- Adjacency (AssociationStore)
- Route Evidence
- Primitive登録

**Replay**: 既存Experienceを内部で再処理。

更新対象:
- Chunk familiarity (use_count / usage_strength)
- Representation (predict edges)
- Shortcut候補 (merge_candidates)
- 処理効率

**非更新 (Replay時):**
- `metrics.total_characters` / `metrics.total_decisions`（外部観測統計を水増ししない）
- `AssociationStore`（Adjacencyは外部共起のみ）
- Primitive登録（新primitiveはExperience時のみ）

**原則**: 同じ文章を8回Replayしても、世界で8回観測したとはみなさない。

**現行Adaptive TrainerはExperience/Replayを区別しないため移行対象 (Phase 1)。**

API:
```rust
model.expose_external(text)  // Experience
model.replay(text)           // Replay
model.expose(text)           // expose_external() へのalias (後方互換)
```

---

## 22. Feedback

**状態: 確定・実装済**

UsageとCorrectnessを分離。

- `used ≠ correct`
- Positive / Negative / Avoidanceを保持
- NegativeはContent自体を削除せず、Context-dependent Avoidanceとして扱う

---

## 23. Turn Learning

**状態: 確定・実装済**

```
State
→ Frozen Generation
→ Trace
→ External Exposure
→ Optional Feedback
→ Next State
```

- 生成中にモデル変更しない
- 自己生成出力を自動Exposureしない

---

## 31. Cycle Consistency

**状態: 候補だが有力**

```
T_{A→B} · T_{B→C} · T_{C→A} ≈ 1
log-domain: t_{A→B} + t_{B→C} + t_{C→A} ≈ 0
```

局所Relationの整合性評価へ利用する。全グラフcycle検査は行わない。

---

## 32. Substitution Consistency

**状態: 候補だが有力**

二つのViewを置換しても周辺Relation構造が保たれるほど、同一Identity Evidenceを上げる。IdentityとCorrelationを区別する主要手掛かり。
