from utils import brachistochrone

if __name__ == '__main__':
    config = brachistochrone.load_boundary_conds('res/cfg/boundary-conds.json')
    brachistochrone.pretty_print_constants(config)
    brachistochrone.compare()
