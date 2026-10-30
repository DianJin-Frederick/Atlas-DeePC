# Atlas-DeePC

Research code and the project website are maintained separately:

```text
atlas_deepc/
  code/       Controllers, notebooks, local offline datasets, and experiment links
  website/    Research website and its published assets
  .git/       Shared repository history
```

- [Research code and experiments](code/README.md)
- [Notebooks](code/notebooks/README.md)
- [Website](website/README.md)

Run research commands from `code/`. Preview the website from this root with
`python3 -m http.server 8000 --bind 127.0.0.1 --directory website/dist`.

The Select-DPC comparison uses a pinned upstream dependency. After cloning, run
`git submodule update --init --recursive` from this root to fetch it. The
F1TENTH simulator is vendored with its license in `code/third_party/f1tenth_gym/`.

GitHub stores code, website assets, essential offline inputs, and saved summaries.
Full rollout arrays, experiment videos, local environments, and historical archives
are excluded. Some detailed saved-result plots require the separately stored data;
this private working repository is not yet a standalone publication package.

Large saved experiment files live on the portable SSD at
`/Volumes/Crucial X10/atlas_deepc_data/`; the original offline datasets and compact
summaries remain in `code/`. Connect the SSD before loading raw rollouts or atlas
caches. Existing file paths are preserved through links in `code/experiments/`.

The old `/Users/kumiko/Projects/atlas_deepc_reacher` location is a compatibility
link to this project's `code/`, not a second copy. Historical result manifests
and saved notebook outputs still use that path. Open this `atlas_deepc` folder
for new work; restart notebook kernels after the move.
