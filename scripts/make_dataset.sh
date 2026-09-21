python dataset/make_dataset.py \
    --img_root /home/syk/pest_cls_dataset/imgs/ \
    --dataset_csv /home/syk/pest_cls_dataset/annotations.csv \
    --class_map_json /home/syk/pest_cls_dataset/class_map_merge.json \
    --taxonomy_csv /home/syk/pest_cls_dataset/captions/pest_filted.csv \
    --caption_jsons /home/syk/pest_cls_dataset/captions/pest_corpus_definitions.json \
    --out_dir /home/syk/pest_cls_dataset/captions/shorter \
    --ratios 0.7 0.3 0 \
    --num_captions 3 \
    --contrast_prob 0 \
    --simple_prob 0.3 \
    --min_dims 1 \
    --max_dims 2 \
    --seed 1916161
    # --make_lmdb \
    # --lmdb_out /home/syk/pest_cls_dataset/captions/multimodal.lmdb \
