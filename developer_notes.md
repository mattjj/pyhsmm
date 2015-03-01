Some notes on current build/compile pipeline:


If you're a user of a stable release 
----------------------------------------------

You simply do:

``pip install pyhsmm``

and not worry about Cython or anything else.

If you're a developer
----------------------------

You edit the Cython code and then compile it using either:

```
python setup.py install --with-cython
```

or:

```
python setup.py build --with-cython
```

If you're using using virtual environments and ``pip`` and want
``pyhsmm`` installed in your virtual environment in editable mode,
run:

```
pip install --edit .
```

If you're a developer who is making and testing a stable release
---------------------------------------------------------------------

Get your code in a state you like, and then wipe out all the Cython-generated files with:

```
python setup.py sdist 
````

This will make a source distribution in ``dist/``. Untar that somewhere fresh and test it using:

```
cd pyhsmm-0.1.x/
pip install --edit .
```

This should trigger compilation of *.cpp files **without** invoking Cython or requiring it to be installed on the machine where you're installing the pyhsmm release.
