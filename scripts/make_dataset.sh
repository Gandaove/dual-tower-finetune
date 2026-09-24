python -m dataset.make_dataset \
    --img_root /home/syk/pest_cls_dataset/imgs/ \
    --dataset_csv /home/syk/pest_cls_dataset/annotations-xy.csv \
    --class_map_json /home/syk/pest_cls_dataset/class_map_merge.json \
    --taxonomy_csv /home/syk/pest_cls_dataset/captions/taxonomy.csv \
    --caption_jsons /home/syk/pest_cls_dataset/captions/pest-corpus-build.json \
    --out_dir /home/syk/pest_cls_dataset/captions/v2 \
    --ratios 0.7 0.3 0 \
    --num_captions 3 \
    --contrast_prob 0.15 \
    --simple_prob 0.2 \
    --min_dims 1 \
    --max_dims 3 \
    --confusion_json outputs/fgclip2-20260923/valid1/confusion_intensity.json \
    --hard_prob 0.8 \
    --seed 13894854
    # --make_lmdb \
    # --lmdb_out /home/syk/pest_cls_dataset/captions/multimodal.lmdb \
