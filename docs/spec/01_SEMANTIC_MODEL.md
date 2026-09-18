# 01 Semantic Model

## 1. 最上位目的

**状態: 確定**

SynTrail-LMの目的:

> 経験によって、以前必要だった探索・判断・計算を再利用可能な構造へ変換し、
> 同じ又は類似する処理を次回以降より少ない計算で実行すること。

基本循環:

```
未知 → 細粒度処理 → 探索・判断 → 成功・反復
  → 構造化 → 高次化 → Shortcut → 次回の計算削減
```

使われなくなった構造:

```
Shortcut → 弱化 → SLEEP → 必要なら旧RepresentationへFallback
```

**定義: 学習 = 情報量を増やすだけでなく、再計算を減らすこと**

---

## 2. Content

**状態: 確定**

SynTrail内部で処理対象となる任意の情報単位。

例: Unicode文字 / 文字列 / Chunk / 単語 / 文 / 発話 / 状況 / 概念 / Skill / Route Pattern / 高次構造

文字→単語→文→概念という固定階層を事前定義しない。Contentは意味上の上位概念であり、必ずしも単一の物理structを要求しない。

---

## 3. Primitive

**状態: 確定・実装済**

- 最小入力単位はUnicode Scalar Value
- 1 scalar = 1 Primitive
- Stable IDを持つ
- 原則削除しない
- Chunk等が利用できない場合の最終Fallback
- UTF-8 byteは意味上のPrimitiveではない

---

## 4. Chunk

**状態: 確定・実装済**

複数Unitを再利用可能な一単位へ圧縮したもの。

```
ChunkCore { left, right }
```

Chunk自身も別Chunkの子になれる。Primitive → Chunk → Chunk of Chunks → ... と再帰化できる。

---

## 5. Identity

**状態: 確定・未実装**

同じ対象として扱われる複数Viewを束ねる最小Anchor。

- Identity自身へ大量の知識を直接詰め込まない
- `Identity ≈ address / anchor`
- 自己変換は恒等: `T_{A→A} = 1`（自己Edgeの物理保存は不要）

---

## 6. View

**状態: 確定・未実装**

一つのIdentityが特定の観測・表現・処理方法からどう現れるかを表す。

**内部View**: 同一Contentの異なる分割粒度
```
ABCDE / A/B/C/D/E / A/BC/D/E / AB/CDE
```

**外部View**: 異なる表層表現
```
DOG-ID / 犬 / dog / 🐕
```

---

## 7. Representation

**状態: 確定・未実装**

ViewのうちContentをどのUnit列・処理粒度で扱うかを表す内部処理View。

```
R0 = [A][B][C][D][E]
R1 = [A][BC][D][E]
R2 = [AB][CDE]
R3 = [ABCDE]
```

獲得順と使用優先順位を分離する。通常は「十分信頼できる候補のうち処理コストが低いもの」を優先する。

---

## 8. Exact Identity と Cross-View Identity

**Exact Identity (確定・部分実装)**

展開Primitive系列が完全一致するもの (`AB/CDE`, `A/BCDE`, `ABCDE` など)。

**Cross-View Identity (確定・未実装)**

表層は異なるが、Context・置換可能性・整合性から同じ対象へ収束するもの (`犬`, `dog`, `🐕`)。

制約: `犬`, `柴犬`, `ドッグフード`, `犬の写真` まで一つのIdentityへまとめない。

旧仕様「展開Primitive系列が一致すれば同一」は残すが、Identity全体の一部とする。

---

## 9. Transform

**状態: 確定方針・未実装**

View間・Identity間の視点変換は可逆Transformとして扱える。

```
T_{B→A} = T_{A→B}^{-1}
合成: T_{A→C} = T_{A→B} · T_{B→C}
```

log-domainなら:
```
t_{A→A} = 0
t_{B→A} = -t_{A→B}
t_{A→C} = t_{A→B} + t_{B→C}
```

**注意**: ID番号自体を逆数にしない。IDは対象、Transformは視点変換。

---

## 10. Transform と Evidence の分離

**状態: 確定**

Relationには最低でも意味上以下を区別できること:

- Transform
- Evidence
- Directional Usage
- Recency
- Uncertainty

ただし全てを毎Relationへ物理保存するとは限らない。導出可能な値は保存しない。

---

## 11. Relation

**状態: 確定アーキテクチャ**

Relationは共通の疎関係基盤を持つ。

**主要な意味分類:**
- View membership / Adjacency / Recall / Route / Lineage / subtype / related / represents / avoidance

**共通操作原理:** source / target / strength/evidence / context / time

意味差まで消さない。詳細は [02_MEMORY_MODEL.md](02_MEMORY_MODEL.md) の各節参照。
