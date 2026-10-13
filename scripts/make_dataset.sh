python -m dataset.make_dataset \
    --img_root /home/syk/pest_cls_dataset/imgs/ \
    --dataset_csv /home/syk/pest_cls_dataset/annotations-xy.csv \
    --class_map_json /home/syk/pest_cls_dataset/class_map_merge.json \
    --taxonomy_csv /home/syk/pest_cls_dataset/captions/taxonomy.csv \
    --caption_jsons /home/syk/pest_cls_dataset/captions/pest-corpus-build.json \
    --anchors_yaml /home/syk/pest_cls_dataset/captions/anchors.yaml \
    --out_dir /home/syk/pest_cls_dataset/captions/v4 \
    --prompt_prefix "A photo of" \
    --ratios 0.7 0.3 0 \
    --num_captions 3 \
    --diagnostic_prob 0.3 \
    --min_dims 1 \
    --max_dims 3 \
    --seed 1891898456
    # --make_lmdb \
    # --lmdb_out /home/syk/pest_cls_dataset/captions/multimodal.lmdb \
