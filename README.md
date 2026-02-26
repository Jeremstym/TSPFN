## How to run

### Install

First, download the project's code:
```shell script
# clone project
git clone url
```

Next you have to install the project and its dependencies. The project's dependency management and packaging is handled
by [`poetry`](https://python-poetry.org/) so the recommended way to install the project is in a virtual environment
(managed by your favorite tool, e.g. `conda`, `virtualenv`, `poetry`, etc.), where
[`poetry` is installed](https://python-poetry.org/docs/#installation). That way, you can simply run the command:
```shell script
poetry install
```

> **Note**
> When a [`poetry.lock`](poetry.lock) file is available in the repository, `poetry install` will automatically use it to
> determine the versions of the packages to install, instead of resolving anew the dependencies in `pyproject.toml`.
> When no `poetry.lock` file is available, the dependencies are resolved from those listed in `pyproject.toml`, and a
> `poetry.lock` is generated automatically as a result.

> **Warning**
> Out-of-the-box, `poetry` offers flexibility on how to install projects. Packages are natively `pip`-installable just
> as with a traditional `setup.py` by simply running `pip install <package>`. However, we recommend using `poetry`
> because of an [issue with `pip`-installing projects with relative path dependencies](https://github.com/python-poetry/poetry/issues/5273)
> (the `vital` submodule is specified using a relative path). When the linked issue gets fixed, the setup instructions
> will be updated to mention the possibility of using `pip install .`, if one wishes to avoid using `poetry` entirely.

### Data

Use data in ```.csv``` format, where the last "column" of each row is the label.
Put each dataset in the same folder that you could target with a ```.env```variable path directory.

You just have to list your ```.csv``` in [config data pretraining-csv.yaml](config/data/pretraining-csv.yaml)

**Warning** 
Insure that each dataset does not contain more than 10 labels !! As we follow the TabPFN architecture (v1) and download its weights, 
we cannot afford more than 10 label classification for now.
