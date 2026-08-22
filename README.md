# Face Recognition

An prototype for one-to-one face verification.

## Fine-tuning ResNet18

The project uses ImageNet-pretrained ResNet18 with a 512-dimensional ArcFace embedding.
Choose the training scope explicitly:

```
uv run --group training python src/training.py --fine-tuning last-layer
```

Use `--fine-tuning all` to update the full backbone. Both modes train the new
embedding layer and ArcFace weights. Training metrics and the final checkpoint
are saved to local MLflow.

Resume an interrupted run with its checkpoint:

```
uv run --group training python src/training.py --fine-tuning last-layer --resume-from checkpoints/resnet18-last-layer.pt
```

Inspect all available options with:

```
uv run --group training python src/training.py --help
```
