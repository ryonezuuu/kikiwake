#!/usr/bin/env bash
# Cowork の Customize > Skills にアップロードするスキルの ZIP（dist/kikiwake-skill.zip）を作る。
# ZIP のルートは kikiwake/ フォルダで、中身は skill/ の内容（kikiwake/SKILL.md ほか）。
# どのディレクトリから実行してもよい。既にある ZIP は作り直す。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
skill_dir="$repo_root/skill"
out_dir="$repo_root/dist"
zip_path="$out_dir/kikiwake-skill.zip"

[ -f "$skill_dir/SKILL.md" ] || { echo "エラー: $skill_dir/SKILL.md が見つかりません" >&2; exit 1; }
command -v zip >/dev/null || { echo "エラー: zip コマンドが見つかりません" >&2; exit 1; }

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

mkdir -p "$work_dir/kikiwake" "$out_dir"
cp -R "$skill_dir/." "$work_dir/kikiwake/"
rm -f "$zip_path"  # zip -r は既存の ZIP に追記するので、先に消して作り直す
(cd "$work_dir" && zip -rqX "$zip_path" kikiwake -x '*.DS_Store')

echo "$zip_path"
