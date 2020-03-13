class BaseDataset(object):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __getitem__(self, index):
        raise NotImplementedError


class IndexedDataset(object):
    r"""An abstract class representing a :class:`Sampler`.

    Used to index datasets in a way specific for the task at hand.
    """

    def __init__(self, base_dataset, transforms):
        self.dataset = base_dataset
        self.T = transforms

    def __getitem__(self, index: int):
        raise NotImplementedError

    def cache(self, ):
        raise NotImplementedError

class Loader(object):
    r"""An abstract class representing a :class:`Loader`.

    Used to load the actual data from filenames provided by an Sampler
    """

    def __init__(self, adds: tuple, requires: tuple = ()):
        self.requires = requires
        self.adds = adds

    def __call__(self, sample: dict) -> dict:
        raise NotImplementedError

class Modifier(object):
    r"""A base clas representing a :class:`Modifier`.

    Used to perform modifications on existring sample elements
    """

    def __init__(self, name: str, keep_originals: bool = True,
            requires: tuple =(), modifies: tuple =(), adds: tuple = ()):
        self.requires = requires
        self.modifies = modifies
        self.adds = adds
        self.name = name
        self.keep_originals = keep_originals

    def _rename(self, sample: dict, field: str) -> str:
        new_name = f"{field}_{self.name}"
        if self.keep_originals:
            sample[new_name] = sample[field].copy()
        else:
            sample[new_name] = sample[field]
        return new_name
        
    def __call__(self, sample: dict, keep_originals: bool) -> dict:
        raise NotImplementedError

class ParameterGenerator(object):
    r"""An abstract class representing a :class:`ParameterGenerator`.
    
    Used to generate (mostly random) parameters for Modifiers
    """

    def __init__(self, adds: tuple, requires: tuple = ()):
        self.requires = requires
        self.adds = adds

    def __call__(self, sample: dict) -> dict:
        raise NotImplementedError
