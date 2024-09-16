# python3 -m src.main +experiment=acid \
# mode=test \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
# checkpointing.load=checkpoints/acid.ckpt

# python3 -m src.main +experiment=re10k \
# mode=test \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
# checkpointing.load=checkpoints/acid.ckpt

python convert_colmap_hs5.py && \


rm -rf outputs/test && \
python3 -m src.main +experiment=7s_scene_fire_n2 \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=datasets/7s/n2/scene_fire/test/evaluation.json \
checkpointing.load=checkpoints/acid.ckpt