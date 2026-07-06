#!/bin/bash

main() {
    DATA_DIR=${DATA_DIR:-"./datasets/AvatarReX/"}

    OUT_DIR="preprocessed_data/avatarrex"
    TEST_JSON="${OUT_DIR}/test.json"

    python preprocess_dataset/preprocess_avatarrex.py \
        --data_root "${DATA_DIR}" || { echo "Preprocessing failed."; return 1; }

    echo "Creating fixed eval indices (1 input view -> 4 target views)..."
    python preprocess_dataset/create_fixed_eval_indices.py \
        --test_json ${TEST_JSON} \
        --output ${OUT_DIR}/fixed_eval_views_avatarrex_1.json \
        --input_views 1 \
        --target_views 4 || { echo "Preprocessing failed."; return 1; }

    echo "Creating fixed eval indices (4 input views -> 4 target views)..."
    python preprocess_dataset/create_fixed_eval_indices.py \
        --test_json ${TEST_JSON} \
        --output ${OUT_DIR}/fixed_eval_views_avatarrex_4.json \
        --input_views 4 \
        --target_views 4 || { echo "Preprocessing failed."; return 1; }

    echo "Preprocessing complete!"
}

main