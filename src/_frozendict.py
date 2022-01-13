from collections import Mapping


class frozendict(Mapping):
    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)

    def __repr__(self):
        return f"frozendict({repr(self._dict)})"

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __getitem__(self, key):
        return self._dict[key]

    def __hash__(self):
        items = self.items()
        immutable_set = frozenset(items)
        return hash(immutable_set)
