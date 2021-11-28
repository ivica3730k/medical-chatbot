# Some code to make private members visible in documentation
__pdoc__ = {name: True
            for name, obj in globals().items()
            if name.startswith('_') and callable(obj)}
