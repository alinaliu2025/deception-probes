"""Registry so scripts can ask for a dataset by name."""

from . import omission, sandbagging, sycophancy

BUILDERS = {
    "sycophancy": sycophancy.build,
    "sandbagging": sandbagging.build,
    "omission": omission.build,
}


def get(deception_type: str):
    if deception_type not in BUILDERS:
        raise KeyError(f"unknown type {deception_type!r}; have {list(BUILDERS)}")
    return BUILDERS[deception_type]()
