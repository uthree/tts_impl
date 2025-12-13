import argparse
import inspect


# 任意の関数
def my_function(name: str, age: int = 25, verbose: bool = False):
    """
    Example function.

    Args:
        name (str): User's name.
        age (int): User's age. Defaults to 25.
        verbose (bool): Enable verbose mode. Defaults to False.
    """
    if verbose:
        print("Verbose mode is enabled.")
    print(f"Name: {name}, Age: {age}")


# generate argument parser automatically
def parse_args_for_function(func):
    parser = argparse.ArgumentParser(description=func.__doc__)
    sig = inspect.signature(func)

    for name, param in sig.parameters.items():
        arg_type = (
            param.annotation if param.annotation != inspect.Parameter.empty else str
        )
        default = param.default

        if default == inspect.Parameter.empty:
            # required arg.
            parser.add_argument(f"{name}", type=arg_type, help=f"{name} (required)")
        else:
            # optional arg.
            parser.add_argument(
                f"--{name}",
                type=arg_type,
                default=default,
                help=f"{name} (default: {default})",
            )

    return parser.parse_args()


# コマンドライン引数をパースして関数を呼び出し
if __name__ == "__main__":
    args = parse_args_for_function(my_function)
    kwargs = vars(args)  # argparse.Namespace -> dict
    my_function(**kwargs)
