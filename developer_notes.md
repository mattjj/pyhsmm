Here's the pipeline as I have it now. It works on my end:

If you're a developer
----------------------------

You edit the Cython code and then compile it using either:

```
python setup.py install --with-cython
```

or:

```
python setup.py sdist
```

The latter also makes the source distribution. 

If you're a developer who is testing a stable release
---------------------------------------------------------------------

Get your code in a state you like, and then wipe out all the Cython-generated files with:

```
python setup.py clean
````

Then run: 

```
python setup.py sdist
````

This will make a source distribution in ``dist/``. Untar that somewhere fresh and test it using:

```
cd pyhsmm-0.1.x/
pip install .
```

This should trigger compilation of *.cpp files **without** invoking Cython or requiring it to be installed on the machine where you're installing the pyhsmm release.

If you're a user of a stable release 
----------------------------------------------

You simply do:

``pip install pyhsmm``

and not worry about Cython or anything else.
