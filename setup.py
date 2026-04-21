from setuptools import setup, find_packages

setup(
    name="latentft",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tqdm",
        "torch",
        "torchaudio",
        "lightning",
        "hydra-core",
        "webdataset",
        "torchvggish",
        # BigVGAN.from_pretrained is incompatible with huggingface_hub>=0.25
        "huggingface_hub>=0.21,<0.25",
        "bigvgan @ git+https://github.com/maswang32/BigVGAN.git",
        "wandb",
    ],
    extras_require={
        "reproduce_results": [
            "librosa",
            "descript-audio-codec",
            "mir_eval",
            "essentia",
        ]
    },
)