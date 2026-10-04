import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: integration tests that play real games; deselect with -m 'not slow'",
    )


@pytest.fixture(autouse=True, scope="session")
def _install_control_table():
    """Hand the W3 control table to the Rust encoder for the whole session.

    Production installs it at the three Rust entry points (`derive_records_rust`,
    the flat batch adapters, the advisor's searcher), but tests reach
    `seven_wonders_rust` directly and would otherwise hit the deliberate panic.
    The panic is the right production behaviour -- a missing table must stop the
    run rather than emit zeros that read as "the opponent reaches every slot
    first" -- so this fixture is a test convenience, not a relaxation of it.
    """

    try:
        from .control_table import ensure_rust_table
    except Exception:
        return
    ensure_rust_table()


@pytest.fixture(autouse=True)
def _exact_tactics_off():
    """Every test starts with G4's process-wide switch OFF.

    Constructing an advisor (or running `phase_d.main`) turns it on for the
    whole process, and the exact Python/Rust equivalence gates assume plain
    search. Tests that want it on set it themselves.
    """

    try:
        import seven_wonders_rust as swr
    except ImportError:
        yield
        return
    swr.set_exact_tactics(False)
    yield
    swr.set_exact_tactics(False)
