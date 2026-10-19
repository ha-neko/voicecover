from __future__ import annotations

from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    RichLog,
    Select,
)

from .paths import default_output
from .pipeline import Options, run


class PipelineProgress(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class PipelineFinished(Message):
    def __init__(self, output: Path | None = None, error: str | None = None) -> None:
        super().__init__()
        self.output = output
        self.error = error


class VoicecoverApp(App[None]):
    TITLE = "Voicecover"
    SUB_TITLE = "video translation and dubbing"
    CSS_PATH = "tui.tcss"
    BINDINGS = [("ctrl+q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="form"):
            yield Label("Video source", classes="field-label")
            yield Input(placeholder="Local video path or YouTube URL", id="input")
            yield Label("Output file", classes="field-label")
            yield Input(placeholder="Automatic .mp4 path", id="output")
            with Horizontal(classes="row"):
                with VerticalScroll(classes="field"):
                    yield Label("Target language", classes="field-label")
                    yield Input(placeholder="es, fr, ja, pt-BR...", id="target")
                with VerticalScroll(classes="field"):
                    yield Label("Source language", classes="field-label")
                    yield Input(placeholder="Auto-detect", id="source")
            yield Label("Translation provider", classes="field-label")
            yield Select(
                (
                    ("Offline NLLB — higher accuracy", "nllb"),
                    ("Offline Argos — smaller", "argos"),
                    ("Google — unofficial", "google"),
                ),
                value="nllb",
                allow_blank=False,
                id="translator",
            )
            yield Label("Whisper context", classes="field-label")
            yield Input(
                placeholder="Optional topic and vocabulary hints",
                id="initial-prompt",
            )
            with Horizontal(classes="row"):
                with VerticalScroll(classes="field"):
                    yield Label("Whisper model", classes="field-label")
                    yield Select(
                        ((name, name) for name in ("tiny", "base", "small", "medium", "large-v3")),
                        value="small",
                        allow_blank=False,
                        id="model",
                    )
                with VerticalScroll(classes="field"):
                    yield Label("Compute device", classes="field-label")
                    yield Select(
                        ((name, name) for name in ("auto", "cpu", "cuda")),
                        value="cpu",
                        allow_blank=False,
                        id="device",
                    )
            with Horizontal(classes="row"):
                with VerticalScroll(classes="field"):
                    yield Label("TTS voice", classes="field-label")
                    yield Input(placeholder="Select automatically", id="voice")
                with VerticalScroll(classes="field"):
                    yield Label("Original audio volume", classes="field-label")
                    yield Input("0.15", type="number", id="volume")
            with Horizontal(id="switches"):
                yield Checkbox("Write translated subtitles", value=False, id="subtitles")
                yield Checkbox("Natural speech grouping", value=True, id="natural-speech")
                yield Checkbox("Keep temporary files", id="keep-temp")
            with Horizontal(id="actions"):
                yield Button("Create dubbed video", variant="primary", id="start")
                yield LoadingIndicator(id="loading")
            yield RichLog(id="log", markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#loading", LoadingIndicator).display = False
        self.query_one("#input", Input).focus()

    def _options_from_form(self) -> Options:
        input_value = self.query_one("#input", Input).value.strip()
        target = self.query_one("#target", Input).value.strip()
        if not input_value:
            raise ValueError("enter a local video path or YouTube URL")
        if not target:
            raise ValueError("enter a target language")

        output_value = self.query_one("#output", Input).value.strip()
        source = self.query_one("#source", Input).value.strip() or None
        voice = self.query_one("#voice", Input).value.strip() or None
        initial_prompt = self.query_one("#initial-prompt", Input).value.strip() or None
        volume_text = self.query_one("#volume", Input).value.strip()
        try:
            volume = float(volume_text)
        except ValueError as exc:
            raise ValueError("original audio volume must be a number from 0 to 1") from exc

        return Options(
            input=input_value,
            output=Path(output_value) if output_value else default_output(input_value, target),
            target_language=target,
            source_language=source,
            model=str(self.query_one("#model", Select).value),
            device=str(self.query_one("#device", Select).value),
            voice=voice,
            original_volume=volume,
            write_subtitles=self.query_one("#subtitles", Checkbox).value,
            keep_temp=self.query_one("#keep-temp", Checkbox).value,
            translator=str(self.query_one("#translator", Select).value),
            initial_prompt=initial_prompt,
            natural_speech=self.query_one("#natural-speech", Checkbox).value,
        )

    def _set_running(self, running: bool) -> None:
        self.query_one("#start", Button).disabled = running
        self.query_one("#loading", LoadingIndicator).display = running

    @on(Button.Pressed, "#start")
    def start_pipeline(self) -> None:
        try:
            options = self._options_from_form()
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        log = self.query_one("#log", RichLog)
        log.clear()
        log.write("[bold cyan]starting voicecover[/bold cyan]")
        self._set_running(True)
        self.run_pipeline(options)

    @work(thread=True, exclusive=True, exit_on_error=False)
    def run_pipeline(self, options: Options) -> None:
        try:
            output = run(options, progress=lambda text: self.post_message(PipelineProgress(text)))
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self.post_message(PipelineFinished(error=f"{exc}\n\n{tb}"))
        else:
            self.post_message(PipelineFinished(output=output))

    @on(PipelineProgress)
    def show_progress(self, message: PipelineProgress) -> None:
        self.query_one("#log", RichLog).write(message.text)

    @on(PipelineFinished)
    def finish_pipeline(self, message: PipelineFinished) -> None:
        self._set_running(False)
        log = self.query_one("#log", RichLog)
        if message.error:
            log.write(f"[bold red]error:[/bold red] {message.error}")
            self.notify(message.error, severity="error", timeout=8)
        else:
            log.write(f"[bold green]complete:[/bold green] {message.output}")
            self.notify(f"Dubbed video created: {message.output}", timeout=8)


def main() -> None:
    VoicecoverApp().run()


if __name__ == "__main__":
    main()
