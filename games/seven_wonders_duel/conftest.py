def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: integration tests that play real games; deselect with -m 'not slow'",
    )
