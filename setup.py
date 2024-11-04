from setuptools import setup, find_packages

setup(
    name="ohara",
    version="0.1",
    packages=find_packages(),
    author="joey00072",
    author_email="00shxf@gmail.com",
    description="A collection of implementations of LLM, papers, and other models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/joey00072/ohara",
    install_requires=[
        # Add your project dependencies here
        # TODO: add requirements here
        "lightning",
        "transformers",
        "datasets",
        "wandb",
        "rich",
        "tensorboard",
    ],
    entry_points={
        "console_scripts": [
            "ohara=ohara.main:main",  # Adjust 'ohara.main:main' as necessary
        ],
    },
    python_requires=">=3.6",
)
