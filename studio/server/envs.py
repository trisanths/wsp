"""Build the env for a run or a builder session, with the Modal backend when the config asks for it."""

from wsp.swarm import factory


def build_run(config):
    """Like factory.build_run. For backend "modal" the studio's ModalCompileEval is the backend class."""
    if config["backend"] == "modal":
        from studio.server import modal_backend

        modal_backend.register()
    return factory.build_run(config)
