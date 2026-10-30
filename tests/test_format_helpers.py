from odtools.conversions.format_helpers import FORMAT_HELPERS


class TestFormatLoaders:
    def test_every_loader_declares_its_format(self):
        for format_type, loader in FORMAT_HELPERS.items():
            assert loader.FORMAT_TYPE is format_type
