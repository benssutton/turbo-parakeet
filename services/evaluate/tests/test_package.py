def test_dependencies_import():
    import chbench.harness.orchestrate  # noqa: F401
    import synthdata.scenarios  # noqa: F401
    import turbo_parakeet.recommenders  # noqa: F401

    import evaluate  # noqa: F401
