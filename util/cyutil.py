import Cython.Build
from Cython.Build.Dependencies import *

# NOTE: mostly a copy of cython's create_extension_list except for the lines
# surrounded by "begin matt added" / "end matt added"
def create_extension_list(patterns, exclude=[], ctx=None, aliases=None, quiet=False, language=None,
                          exclude_failures=False):
    if not isinstance(patterns, (list, tuple)):
        patterns = [patterns]
    explicit_modules = set([m.name for m in patterns if isinstance(m, Extension)])
    seen = set()
    deps = create_dependency_tree(ctx, quiet=quiet)
    to_exclude = set()
    if not isinstance(exclude, list):
        exclude = [exclude]
    for pattern in exclude:
        to_exclude.update(map(os.path.abspath, extended_iglob(pattern)))

    module_list = []
    for pattern in patterns:
        if isinstance(pattern, str):
            filepattern = pattern
            template = None
            name = '*'
            base = None
            exn_type = Extension
            ext_language = language
        elif isinstance(pattern, Extension):
            for filepattern in pattern.sources:
                if os.path.splitext(filepattern)[1] in ('.py', '.pyx'):
                    break
            else:
                # ignore non-cython modules
                module_list.append(pattern)
                continue
            template = pattern
            name = template.name
            base = DistutilsInfo(exn=template)
            exn_type = template.__class__
            ext_language = None  # do not override whatever the Extension says
        else:
            raise TypeError(pattern)

        for file in extended_iglob(filepattern):
            if os.path.abspath(file) in to_exclude:
                continue
            pkg = deps.package(file)
            if '*' in name:
                module_name = deps.fully_qualified_name(file)
                if module_name in explicit_modules:
                    continue
            else:
                module_name = name

            if module_name not in seen:
                try:
                    kwds = deps.distutils_info(file, aliases, base).values
                except Exception:
                    if exclude_failures:
                        continue
                    raise
                if base is not None:
                    for key, value in base.values.items():
                        if key not in kwds:
                            kwds[key] = value

                sources = [file]
                if template is not None:
                    sources += [m for m in template.sources if m != filepattern]
                if 'sources' in kwds:
                    # allow users to add .c files etc.
                    for source in kwds['sources']:
                        source = encode_filename_in_py2(source)
                        if source not in sources:
                            sources.append(source)
                    del kwds['sources']
                if 'depends' in kwds:
                    depends = resolve_depends(kwds['depends'], (kwds.get('include_dirs') or []) + [find_root_package_dir(file)])
                    if template is not None:
                        # Always include everything from the template.
                        depends = list(set(template.depends).union(set(depends)))
                    kwds['depends'] = depends

                if ext_language and 'language' not in kwds:
                    kwds['language'] = ext_language

                # NOTE: begin matt added
                if 'name' in kwds:
                    module_name = str(kwds['name'])
                    del kwds['name']
                else:
                    module_name = os.path.splitext(file)[0].replace('/','.')
                # NOTE: end matt added
                module_list.append(exn_type(
                        name=module_name,
                        sources=sources,
                        **kwds))
                m = module_list[-1]
                seen.add(name)
    return module_list

true_cythonize = Cython.Build.cythonize
true_create_extension_list = Cython.Build.Dependencies.create_extension_list

def cythonize(*args,**kwargs):
    Cython.Build.Dependencies.create_extension_list = create_extension_list
    out = true_cythonize(*args,**kwargs)
    Cython.Build.Dependencies.create_extension_list = true_create_extension_list
    return out

