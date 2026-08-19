# Face Recognition Authentication

An prototype for one-to-one face verification. Users will enroll with three
face images and later authenticate with an email address and a new face image.

## Fine-tuning ResNet50

The project uses ImageNet-pretrained ResNet50 with a 512-dimensional ArcFace embedding.
Choose the training scope explicitly:

```powershell
uv run --group training python src/training.py --fine-tuning last-layer --deterministic
```

Use `--fine-tuning all` to update the full backbone. Checkpoints are saved as
`checkpoints/resnet50-last-layer.pt` or `checkpoints/resnet50-all.pt`.
