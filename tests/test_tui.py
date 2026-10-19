from textual.widgets import Button, Input, LoadingIndicator

from voicecover.tui import VoicecoverApp


async def test_tui_mounts_and_validates_empty_form() -> None:
    app = VoicecoverApp()
    async with app.run_test(size=(100, 55)) as pilot:
        assert app.query_one("#input", Input).has_focus
        await pilot.click("#start")
        await pilot.pause()
        assert not app.query_one("#start", Button).disabled
        assert not app.query_one("#loading", LoadingIndicator).display


async def test_tui_builds_options() -> None:
    app = VoicecoverApp()
    async with app.run_test(size=(100, 55)):
        app.query_one("#input", Input).value = "https://youtu.be/abc"
        app.query_one("#target", Input).value = "es"
        options = app._options_from_form()
        assert options.output.name == "youtube.es.mp4"
        assert options.target_language == "es"
