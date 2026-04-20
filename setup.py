from setuptools import setup, find_packages

setup(
    name="latentft",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22",
        "tqdm",
        "torch",
        "torchaudio",
        "lightning",
        "hydra-core",
        "webdataset",
        "torchvggish",
        "huggingface_hub>=0.21,<0.25",
        "bigvgan @ git+https://github.com/maswang32/BigVGAN.git",
        "wandb",
    ],
    extras_require={
        "scoreq": ["scoreq[gpu]"],
        "reproduce_results": [
            "librosa",
            "descript-audio-codec",
            "mir_eval",
            "essentia",
        ]
    },
)
