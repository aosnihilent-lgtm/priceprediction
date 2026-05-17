# import tomllib

# def load_config(path = "config/pricing_config.toml"):
#     with open(path, 'rb') as f:
#         config = tomllib.load(f)
#     return config

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_config(path="config/pricing_config.toml"):
    with open(path, "rb") as f:
        config = tomllib.load(f)
    return config