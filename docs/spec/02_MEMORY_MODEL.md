# 02 Logical Memory Model

## 12. Adjacency

**状態: 確定・未実装専用化**

局所的な前後・近傍関係。

```
犬 → は
東京 → は
```

現行AssociationStoreは実質的にこの役割に近い。将来はRelation Coreへ統合する。

---

## 13. Recall

**状態: 確定概念・部分実装**

非局所的な「Xと言えば何が浮かぶか」を表す。

```
犬 → { 散歩, 動物, ペット }
```

原則 `source → sparse Top-K` で取得。全Content類似度計算を行わない。

---

## 14. Recognition

**状態: 確定・未実装**

提示されたViewからIdentityを認識する処理。

```
View → Identity candidates
```

RecallよりRecognitionが成功しやすい状態を許容する。これにより「思い出せないが、言われれば分かる」を表現可能とする。

---

## 15. Route

**状態: 確定概念・現行暫定実装**

現在Contextから次の処理・出力・行動へ進む方向付きRelation。

対象: 次Unit / 会話応答 / 推論 / 行動 / Skill / 手順

現行PredictionStoreは一次Unit→next Unit。最終的には `RouteBank[source] → ranked local candidates` とする。

---

## 16. Lineage / Acquisition History

**状態: 確定・未実装**

Representation/Viewの獲得史を保存する。

```
R0 → R1 → R2 → R3
R1 derived_from R0
R2 derived_from R1
```

履歴は単なる配列ではなくRelationとして扱う。必要ならDAGを許容。

---

## 17. Fallback

**状態: 確定・未実装**

高次Viewが失敗した場合に過去の細粒度処理へ戻る。

```
R3 → R2 → R1 → R0
```

専用System 2モデルは必須としない。Lineageの逆走を主要Fallback機構とする。

---

## 18. Shortcut

**状態: 確定思想・高度版未実装**

**Representation Shortcut**: 高次Viewを直接利用。

**Route Shortcut**: `A → B → C → D` を反復後 `A → D` 相当にcompile。

共通原理: 安定した反復計算を直接参照へ置換する。

---

## Memory Architecture（全体方針）

**状態: 確定方針**

「短期記憶」「長期記憶」等を巨大な専用モジュールとして作らない。以下の軸の組合せから記憶状態を生じさせる:

Access Cost / Detail / Confidence / Recency / Search Depth / Residency / Externality

| 現象 | 構造 |
|------|------|
| 即答 | HOT Shortcut |
| 直近記憶 | Recent Overlay |
| 長期記憶 | 圧縮済みIdentity/Relation |
| 少し考える | 数hop探索 |
| 断片想起 | 一部Viewのみ活動 |
| 言われれば分かる | Recognition成功 |
| 聞き覚え | 弱いIdentity候補 |
| 思い出せない何か | 住所・痕跡のみ |
| 手元メモ | 高速External Pointer |
| 外部記憶 | File/DB/Web等 |
