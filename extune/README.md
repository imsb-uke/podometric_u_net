# extune

`extune` (from [`sacred`][sacred]'s "`ex`" convention and [`ray`][ray]'s [`tune`][tune] submodule) is
meant to be a simple demonstration of how to wrap `sacred` experiments with
`tune` for hyper-parameter optimization. This simple example consists of four
directories:

1. `data`: where data is stored
2. `experiments`: where `sacred`'s `FileStorageObserver` will save information.
3. `jupyter`: a place for jupyter notebooks to play around
4. `project`: a python module further divided into four submodules:
  - `model`: code related to the model you want to train
  - `sacred`: decoupled [`sacred`][sacred] code for running and saving information about your experiments
  - `tune`: decoupled [`tune`][tune] code for training the [`sacred`][sacred] code
  - `utils`: place to put helper functions

further, it is not a tutorial on [`sacred`][sacred] or [`tune`][tune]. Both
libraries have good documentation and is worth a read.

# Set up

1. clone the repository
2. `cd <path-to>/extune`
3. `pip install -e ./`
4. `jupyter-notebook jupyter/`
5. open "`Using Tune with Sacred.ipynb`"




[sacred]: https://github.com/IDSIA/sacred
[tune]: https://github.com/ray-project/ray/tree/master/python/ray/tune  
[ray]: https://github.com/ray-project/ray
