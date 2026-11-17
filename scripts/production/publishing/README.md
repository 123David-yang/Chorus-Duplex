# Public Git history check

Before publishing a source commit, inspect the candidate history from the project root.

```sh
python -m scripts.production.publishing.public_guard <candidate-commit>
```

The check reports local-only paths and common credential patterns without printing matching content. It examines every reachable commit, including files deleted later. Blobs over 16 MiB are reported as unscanned. Binary contents and credentials outside the known patterns require separate review.

The working tree's private history contains tracked internal reports. An ignore rule does not remove tracked files or previous commits. Build public commits on the reviewed public history and check the resulting commit before pushing.
