#!/bin/zsh

set -u

project_root="/Users/kumiko/Projects/atlas_deepc_reacher"
source_root="$project_root/experiments/routed_deepc_iid/contextual_cluster_dense_rank_1000"
backup_root="$project_root/rank_ablation_compact_backup"

mkdir -p "$backup_root"

while true; do
  if [[ -d "$source_root" ]]; then
    for rank_dir in "$source_root"/rank_*; do
      [[ -d "$rank_dir" ]] || continue
      rank_name="${rank_dir:t}"
      mkdir -p "$backup_root/$rank_name"
      for compact_file in compact_results.jsonl compact_results_metadata.json summary.json; do
        if [[ -f "$rank_dir/$compact_file" ]]; then
          cp "$rank_dir/$compact_file" "$backup_root/$rank_name/$compact_file.tmp"
          mv "$backup_root/$rank_name/$compact_file.tmp" "$backup_root/$rank_name/$compact_file"
        fi
      done
    done
  fi
  sleep 30
done
