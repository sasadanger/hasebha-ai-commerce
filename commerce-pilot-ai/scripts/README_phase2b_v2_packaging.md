# Phase 2B Correction V2 packaging script

`package_phase2b_correction_v2.ps1` is packaging-only. It hashes protected files byte-for-byte without parsing Test predictions or Test metric contents, creates a fixed allowlist, constructs the V2 ZIP once with fixed member timestamps, and refuses to overwrite an existing V2 ZIP or modify V1 evidence.

It must not be repurposed for model training, scoring, prediction regeneration, or Test evaluation.
