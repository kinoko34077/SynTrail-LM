SynTrail MVP 初期学習データ

目的:
- Unicode文字 → 動的Chunk形成 → Tier昇格 → 生成、の最小検証用。
- 日本語一般性能を測るためのコーパスではありません。
- 同じ部分文字列・語尾・文型を意図的に反復し、Chunkが形成されやすくしています。

ファイル:
- train_control.txt : 学習用
- eval_holdout.txt  : 学習に使わず、形成されたChunkの再利用確認用

推奨確認:
1. 初期状態ではPrimitiveのみ。
2. 学習後に「です」「です。」「今日は」「あります。」等がChunk化するか。
3. Tier 0 → 1 → 2 の昇格が起きるか。
4. eval_holdout.txtで decision/character が学習前より低下するか。
5. 完全文丸暗記だけでなく、未学習文でも既存Chunkが再利用されるか。

文字コード:
UTF-8
