#!/usr/bin/env bash

set -u

expected=824
checked=0
mismatches=0

while IFS= read -r -d '' dir; do
    checked=$((checked + 1))
    count=$(find "$dir" -maxdepth 1 -type f -iname '*.wav' -printf . | wc -c)

    if [[ "$count" -eq "$expected" ]]; then
        printf 'OK\t%s\n' "$dir"
    else
        printf 'MISMATCH\t%s\t%s WAV files (expected %s)\n' \
            "$dir" "$count" "$expected"
        mismatches=$((mismatches + 1))
    fi
done < <(find . -mindepth 1 -maxdepth 1 -type d ! -name '.*' -print0)

printf '\nChecked: %s directories\nMismatches: %s\n' "$checked" "$mismatches"

if [[ "$mismatches" -ne 0 ]]; then
    exit 1
fi
