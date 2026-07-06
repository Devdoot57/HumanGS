#!/bin/bash

main() {
    SCAN_DIR=${SCAN_DIR:-"./datasets/THuman2.1/data/model/"}
    SMPLX_DIR=${SMPLX_DIR:-"./datasets/THuman2.1/data/smplx/"}

    OUT_DIR="preprocessed_data/thuman21"
    TEST_JSON="${OUT_DIR}/test.json"

    python preprocess_dataset/preprocess_thuman21.py \
        --scan_dir "${SCAN_DIR}" \
        --smplx_dir "${SMPLX_DIR}" || { echo "Preprocessing failed."; return 1; }

    echo "Creating fixed eval indices (1 input view -> 4 target views)..."
    python preprocess_dataset/create_fixed_eval_indices.py \
        --test_json ${TEST_JSON} \
        --output ${OUT_DIR}/fixed_eval_views_thuman21_1.json \
        --input_views 1 \
        --target_views 4 || { echo "Preprocessing failed."; return 1; }

    echo "Creating fixed eval indices (4 input views -> 4 target views)..."
    python preprocess_dataset/create_fixed_eval_indices.py \
        --test_json ${TEST_JSON} \
        --output ${OUT_DIR}/fixed_eval_views_thuman21_4.json \
        --input_views 4 \
        --target_views 4 || { echo "Preprocessing failed."; return 1; }

    echo "Preprocessing complete!"
}

main