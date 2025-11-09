import inspect
import os
import types
from pathlib import Path
from typing import Any

import pytest
import ray

TEST_DIR = Path(__file__).parent
RAY_ADDRESS_ENV_VAR = "RAY_ADDRESS"
ENV_VARS = {
    # These are all fake static env. var URIs
    RAY_ADDRESS_ENV_VAR: "ray://fakeray:6789",
    "OPENAI_API_KEY": "sk-fakekey",
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID") or "fakekey",
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY") or "fakesecret",
}


def copy_function(original_function: callable):
    """This functions creates a function copy by accessing the underlying passed
    function internal attributes (i.e.: its code, global vars, name, etc)

    Args:
        original_function (callable): Function to make a copy of

    Returns:
        callable: Resulting copied function
    """
    return types.FunctionType(
        original_function.__code__,
        original_function.__globals__,
        name=original_function.__name__,
        argdefs=original_function.__defaults__,
        closure=original_function.__closure__,
    )


def add_remote_to_method(cls_instance: Any, func: callable, func_name: str) -> None:
    """Overrides the given class' function/method to make it callable with .remote
    to simulate a Ray Actor in the underlying class.

    i.e.: from MyActor.my_func(...) --> MyActor.my_func.remote(...)

    Args:
        cls_instance (Any): Class instance to patch
        func (callable): method or function to add a .remote to
        func_name (str): name of the method or function to patch
    """
    # Make a private copy of the method to keep a reference to the original function.
    # The original function gets overwritten, See below.
    original_function = copy_function(func)

    # Define a function that wraps the original method
    # MyActor.my_func becomes a wrapper which returns a modified 'my_func' with a .remote()
    def func_wrapper(*args, **kwargs):
        return original_function(cls_instance, *args, **kwargs)

    # Set the remote attribute on the wrapper function;
    # Allows us to call 'my_func.remote()'
    func_wrapper.remote = func_wrapper

    # Rewrite the method with the wrapper.
    # It is here where we overwrite and would lose the reference to MyActor.my_func.
    # This step is where we make `my_func.remote()` a class method and therefore
    # MyActor.my_func.remote() callable
    setattr(cls_instance, func_name, func_wrapper)


def add_remote_to_all_methods(cls_instance):
    """Override each method in the class to add `.remote()`"""
    for attr_name in dir(cls_instance):
        attr = getattr(cls_instance, attr_name)
        # We only add .remote() to methods that are not private nor internal
        # so we do not mess up anything python might do internally with the class.
        # We also filter any CLS.remote() so we do not break initialization
        if (
            (inspect.isfunction(attr) or inspect.ismethod(attr))
            and not attr_name.startswith("_")
            and attr_name != "remote"
        ):
            add_remote_to_method(cls_instance, attr, attr_name)


def inject_patched_actor(
    mocker, actor_cls, actor_import_path: str, *actor_init_args, **actor_init_kwargs
):
    """This function patches the import of an Actor anywhere in the codebase by
    injecting a fully patched actor into the namespace.
    """
    # Instantiate an Actor with .remote methods
    model = actor_cls.remote(*actor_init_args, **actor_init_kwargs)
    add_remote_to_all_methods(model)

    remote_mock = mocker.MagicMock()
    remote_mock.remote = mocker.MagicMock(return_value=model)
    clf_mock = mocker.MagicMock()
    clf_mock.remote = remote_mock.remote  # .remote() init
    clf_mock.options = mocker.MagicMock(return_value=remote_mock)  # .options().remote()

    mocker.patch(actor_import_path, new=clf_mock)

    return model


def mock_ray_remote_decorator(*args, **kwargs):
    """Mocks .ray.remote decorator to return the original class with `.options` and `.remote` methods"""

    # NOTE: Any test file importing globally (not within the test function) a Ray remote
    # (i.e.: class or function decorated with ray.remote) or from a file defining a Ray
    # remote can cause this patch to stop working properly.
    # Therefore, Ray remotes should always be imported within the test function.

    # NOTE: Decorating a function or class with '@ray.remote()' (i.e.: empty parenthesis)
    # throws an Assertion error within Ray

    def patch_decorated_remote(cls_or_func):
        """Function to add the `.options()` and `.remote()` methods for class initialization"""
        if inspect.isfunction(cls_or_func):
            # Add the .remote() attribute to the original function
            cls_or_func.remote = cls_or_func
        elif inspect.isclass(cls_or_func):
            # Add the .options() method for class initialization
            # i.e.: Map '<class>.options(...)' --> '<class>' itself
            cls_or_func.options = classmethod(
                lambda cls_or_func, *args, **kwargs: cls_or_func
            )
            # Add the .remote() method for class initialization
            # i.e.: Map '<class>.remote(...)' --> '<class>.__init__(...)'
            cls_or_func.remote = classmethod(
                lambda cls_or_func, *args, **kwargs: cls_or_func(*args, **kwargs)
            )

        return cls_or_func

    # NOTE: This is the first thing that gets executed when python interprets
    # the @ray.remote decorator.
    #
    # First, we check wether we entered because of a decorator with or without parameters.
    # - If '@remote' (1st level decorator) then we directly patch
    #   if it was decorating a function and the wrapper if decorating a class
    #
    # - If '@remote(...)' (2nd level decorator) the decorator was not yet passed the
    # class or function, so we can't do yet the patching and we need to return a
    # regular decorator in charge of doing the patching (i.e.: patch_decorated_remote)
    if len(args) == 1:
        # Here we know the decorator was used without parameters. i.e: @ray.remote
        # and 'args[0]' is the decorated class or function
        if inspect.isfunction(args[0]):
            # We have the decorated function, so patch directly
            args[0].remote = args[0]
            return args[0]
        elif inspect.isclass(args[0]):
            # We have the decorated class, so apply the patch directly
            return patch_decorated_remote(args[0])
    else:
        # Decorator used with parameters. e.g.: @ray.remote(num_cpus=1), therefore
        # at this point we don't have the decorated function or class, we return the
        # patching function instead (this is, behave as a decorator)
        return patch_decorated_remote


@pytest.fixture(autouse=False, scope="module")
def local_ray_for_tests():
    """This fixture runs during the entire session to intialize the connection to Ray"""
    print("⚡ Initializing ray")
    if not ray.is_initialized():
        init_kwargs = {
            "runtime_env": {
                "py_modules": [
                    str(Path.cwd() / "flows"),
                    str(Path.cwd() / "actors"),
                ],
                "working_dir": str(Path.absolute(TEST_DIR)),
            },
        }
        print(init_kwargs)

        # Connect to a local Ray
        # NOTE: For tests we do not provide an address with a protocol
        # so 'ray.init()' should return a 'RayContext' object (local cluster).
        os.environ.pop(RAY_ADDRESS_ENV_VAR, None)
        ray.init(**init_kwargs)

        # Set the RAY_ADDRESS environment variable
        ray_context = ray.runtime_context.get_runtime_context()
        print(f"Ray context type: {type(ray_context)}")
        try:
            os.environ["RAY_ADDRESS"] = ray_context.gcs_address  # RayContext
        except (TypeError, AttributeError):
            os.environ["RAY_ADDRESS"] = ray_context.address  # ClientContext
    yield

    print("🔫 Terminating ray")
    ray.shutdown()


@pytest.fixture(autouse=True, scope="module")
def patch_ray(module_mocker):
    # During tests we create regular classes instead of ActorHandlers. Prefect
    # validates the parameters so we need to allow to pass regular classes to
    # those flows expecting a Ray ActorHandle
    module_mocker.patch("actors.ActorType", new=Any)
    # Makes the @ray.remote decorator return the underlying class instead of an Actor
    module_mocker.patch("ray.remote", new=mock_ray_remote_decorator)
    # Patch ray.init to do nothing
    module_mocker.patch("ray.init", return_value=None)
    # Patch ray.get to pass on the result of the called function
    module_mocker.patch("ray.get", side_effect=lambda x: x)
