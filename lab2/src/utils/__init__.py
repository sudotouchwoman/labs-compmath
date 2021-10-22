def load_boundary_conds(filepath: str) -> dict:
    from json import loads
    with open(filepath, 'r') as confile:
        config = loads(confile.read())
    return config
