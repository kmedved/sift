# SIFT v1.0.2 release — complete; fork detachment awaiting confirmation

Objective: finish PR #108 and publish GitHub-only v1.0.2, verify the uploaded distributions, remove the release branch/worktree, and inspect fork detachment. PyPI publication is excluded.

Completed: PR #108 merged with both original commits intact at `daaffd30cde58e041817601a8737e9fd15a3e35d`; its tree matches tested head `a303c3caca67c04516d12ebe1109fc070aa4a700`. Clean release-source `b95051e2fb31cb0a841c1a1782dd64f7bc5447b3` remains an ancestor. Tag `v1.0.2` points to the merge. Release: https://github.com/kmedved/sift/releases/tag/v1.0.2 . Published notes match the dated 1.0.2 section.

Decisive verification: all six active PR jobs passed in `40340843893`; all eight manual release-gate jobs passed in `40340841371`. All six active merged-main jobs in `40351180658` passed. Release workflow `40351223634` passed, including tag/version matching, builds, Twine, clean-wheel smoke and uploads. Both actual uploaded distributions were downloaded; their SHA-256 digests match GitHub metadata, and all 85 tracked package files in each match the tagged tree. Both pass Twine. The actual published wheel clean-installed outside the checkout and passed `scripts/verify_wheel_install.py`; package and distribution metadata report 1.0.2.

Published artifact SHA-256: wheel `120686542e579029ac3b7b59fc7faf23720bc17780acde09ddc1b3d61c247f5e`; source archive `0316cb71e97f80b3873ead4aa8e07675fcdf3370c8b07fdafd2ad3d12eeeeeb0`. Downloads and fresh verification environment are under `/private/tmp/sift-1.0.2-published`.

Cleanup: the clean, fully merged release worktree and local/remote `claude/1.0.2-release` branches were removed. Only the main checkout and main branch remain. No upstream remote exists.

Release work is complete. Separate pending action: GitHub Settings offers Leave fork network, but warns that detachment is permanent and cannot be reversed; awaiting the user confirmation requested in this chat before executing it. The signed-in in-app browser has that confirmation dialog open.
