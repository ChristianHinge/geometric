torch-model-archiver \
    --model-name geometric \
    --model-file src/models/model.py \
    --version 1.0 \
    --serialized-file model_store/geometric.pt \
    --export-path model_store \
    --handler src/server/graph_handler \
    --force