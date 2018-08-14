import subprocess
import tempfile


def _exec_notebook(path):
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=1000",
                "--output", fout.name, path]
        subprocess.check_call(args)


def test00():
    _exec_notebook('00-matched-filter.ipynb')

def test01():
    _exec_notebook('01-lr_ex.ipynb')

def test02():
    _exec_notebook('02-lr_housing.ipynb')

def test03():
    _exec_notebook('03-bayes_lr_ex.ipynb')

def test04():
    _exec_notebook('04-fisher-example.ipynb')

def test05():
    _exec_notebook('05-pca_ex.ipynb')

def test06():
    _exec_notebook('06-neural-networks-numpy.ipynb')
