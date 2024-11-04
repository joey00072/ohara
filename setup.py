from setuptools import setup, find_packages

setup(
    name="ohara",
    version="0.1.3",
    packages=find_packages(),
    author="joey00072",
    author_email="00shxf@gmail.com",
    description="A collection of implementations of LLM, papers, and other models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/joey00072/ohara",
    install_requires=[
        # Add your project dependencies here
        "lightning>=2.3.0",
        "transformers>=4.46.0",
        "datasets>=3.1.0",
        "wandb>=0.18.0",
        "rich>=13.19.0",
        "tensorboard>=2.18.0",
    ],
    entry_points={
        "console_scripts": [
            "ohara=ohara.main:main",  # Adjust 'ohara.main:main' as necessary
        ],
    },
    python_requires=">=3.6",
)
