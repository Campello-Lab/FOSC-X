def test_import_fosc():
    import foscx
    from foscx import FOSCX

    assert hasattr(foscx, "FOSCX")
    assert callable(FOSCX)
