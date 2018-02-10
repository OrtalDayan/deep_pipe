import os
from resource_manager.token import TokenType

from dpipe.config.base import get_resource_manager
from resource_manager.parser import Array, Dictionary, Literal


def generate_configs(config_path, grid_name):
    rm = get_resource_manager(config_path)
    undef = rm._scopes[0]._undefined_resources
    if grid_name not in undef:
        raise AttributeError(f'"{grid_name}" not present in the config')
    tree = undef[grid_name]
    if type(tree) is not Array or any(type(x) is not Dictionary for x in tree.values):
        raise ValueError(f'{grid_name} is not a list of dictionaries')

    configs = []
    for item in tree.values:
        override = ''
        for name, value in item.pairs:
            if type(name) is not Literal or name.main_token.type != TokenType.STRING:
                raise ValueError('dict keys must be strings')
            name, value = eval(name.to_str(0)), value.to_str(0)
            override += f'\n{name} = {value}'
        rm = get_resource_manager(config_path)
        rm.string_input(override)
        configs.append(rm.render_config())
    return configs


def save_configs(configs, base_path):
    for i, config in enumerate(configs):
        with open(os.path.join(base_path, f'set_{base_path}')) as f:
            f.write(config)


def grid_search(config_path, grid_name, experiment_path):
    configs = generate_configs(config_path, grid_name)
    save_configs(configs, experiment_path)
    # TBC
    # for i in range(len(configs)):


if __name__ == '__main__':
    configs = generate_configs('../../test_GC.config', 'grid')
