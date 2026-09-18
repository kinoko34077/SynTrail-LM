# 04 Execution Model

## 25. HOT / SLEEP

**状態: 論理実装済・物理分離未実装**

**HOT:**
- 通常Fast Path対象
- Active Relation / Strength / Feedback / Recent metadata

**SLEEP:**
- 通常Fast Pathから除外
- 同一IDを維持
- 必要時に復活

---

## 26. Immutable Core / Active Overlay

**状態: 確定・物理未実装**

ChunkやIdentityを `Stable Core + Active Overlay` へ分離。

SLEEPではCore・住所・最低限Relationだけを残し、重いOverlayを外せるようにする。

---

## 27. 疎計算原則（最重要・確定）

通常Inference / Exposureで禁止事項:

- 全Identity走査禁止
- 全Relation走査禁止
- 全候補正規化禁止
- 全世界への反復伝播禁止

一入力で触るのは局所Top-Kのみ。

```
Storage ≈ O(N·K)      N=総Identity数, K=局所候補上限
Fast Path ≈ O(K)
局所整合性 ≈ O(K²) 以下
```

---

## 28. Relation Budget

**状態: 確定方針**

Roleごとに無制限Top-Kを持たない。Identity単位の総Active Budgetを持つ。数値は調整対象。

---

## 29. Lazy Decay

**状態: 確定方針・未実装**

全Relationを毎tick更新しない。`strength` と `last_tick` を保持し、アクセス時に計算:

```
s_now = s_stored · λ^Δt
```

---

## 30. Local Consolidation

**状態: 確定方針**

重い処理をFast Pathから分離。対象:

- consistency / lineage整理 / shortcut compile
- candidate competition / HOT/SLEEP / factorization

変更された局所をdirtyとして処理する。

---

## Micro-ISA

**状態: 確定方針・未実装 (Phase 13)**

Semantic ModelとBackend間へ意味上の最小操作を定義する。

候補命令:
```
ID.RESOLVE    VIEW.GET      VIEW.BIND
REL.GET       REL.BUMP      REL.DECAY
REL.INV       REL.COMPOSE   CHUNK.FIND
CHUNK.MAKE    CHUNK.SLEEP   CHUNK.WAKE
REP.GET       REP.FALLBACK  LINEAGE.PARENT
SELECT        COMPILE       EMIT
```

現時点ではbinary opcodeを固定しない。Micro-ISAをインタプリタ化することも必須ではない。
