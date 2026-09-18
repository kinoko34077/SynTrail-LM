# 05 Physical Model

## 33. Stable ID

**状態: 確定・現行部分実装**

永続参照上のIDを原則詰め直さない。理由:

- Chunk参照 / Relation / Snapshot / Lineage / History を壊さないため

---

## 34. Arena / Runtime Index

**状態: 方向性確定**

Stable IDとは別に、HOTな局所領域へ短いRuntime Indexを割り当てる。

```
Stable ID → (resolve) → Arena Index → Flat Array
```

---

## 35. Variable-width ID

**状態: 候補**

物理保存では `b = ⌈log₂N⌉` に近い幅を使える。

候補: 16bit / 24bit / 32bit / region/block単位幅

Stable ID自体の意味は変えない。

---

## 36. Source-indexed Flat Relation Array

**状態: 有力設計**

現在の `Relation { source, target, ... }` を:

```
RelationIndex[source] → (offset, count)
```

へ変える。Relation record自身からsourceを除去可能。

---

## 37. Packed Binary

**状態: 候補・物理最適化**

論理フィールド数と物理保存フィールド数を分離する。例:

```
target   24bit
strength 16bit
role      3bit
flags     5bit
```

具体幅はprofile・精度試験後に決定。

---

## 38. Fixed Point

**状態: 候補**

Transform / Strength / Evidence等を固定小数点化可能。

特にlog Transformなら:
```
INV = NEG
COMPOSE = ADD
```

整数演算との相性がよい。

---

## 41. Persistence

**状態: 将来設計**

長期的には:

```
in-memory sparse arrays
+ append-only journal
+ periodic snapshot
```

を有力とする。全モデルJSON書換えを永続的な最終形とはしない。

---

## 5層アーキテクチャ（全体像）

```
1. Semantic Model
   Content / Chunk / Identity / View / Relation
       ↓
2. Logical Memory Model
   HOT/SLEEP / Top-K / Lineage / Memory / Shortcut
       ↓
3. Learning & Stabilization
   Experience / Replay / Factorization / Consistency / Feedback
       ↓
4. Execution Model
   Micro-ISA / Fast Path / Local Consolidation
       ↓
5. Physical Model
   Stable ID / Arena Index / Flat Array / Packed Binary
   Fixed Point / Journal/Snapshot
       ↓
Machine Backend
   Rust → SIMD → Assembly → FPGA/ASIC候補
```

---

## 恒久禁止事項

- 全Identity all-pairs Relationを作らない
- 全Relationを通常時に再計算しない
- Replayを外部出現頻度として扱わない
- Identity Objectへ大量状態を詰め込まない
- View同士を全結合しない
- inverseを無条件で二重保存しない
- 全Compositionを事前materializeしない
- 可変bit化のためStable IDを書き換えない
- Micro-ISAを作ることと低速bytecode interpreterを同一視しない
- profile前に手書きAssemblyへ移行しない
