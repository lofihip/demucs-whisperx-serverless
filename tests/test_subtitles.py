from app.utils.subtitles import render_srt, render_tsv, render_txt, render_vtt


def test_render_srt_includes_speaker_labels() -> None:
    output = render_srt([{"start": 0.0, "end": 1.25, "text": "Hello", "speaker": "SPEAKER_00"}])
    assert "00:00:00,000 --> 00:00:01,250" in output
    assert "[SPEAKER_00] Hello" in output


def test_render_vtt_and_txt_are_human_readable() -> None:
    segments = [{"start": 1.0, "end": 2.0, "text": "World", "speaker": "SPEAKER_01"}]
    assert "WEBVTT" in render_vtt(segments)
    assert "[SPEAKER_01] World" in render_txt(segments)


def test_render_tsv_normalizes_tabs_and_newlines() -> None:
    output = render_tsv([{"start": 0.0, "end": 1.0, "speaker": "A", "text": "Hi\tthere\nfriend"}])
    assert output.splitlines()[0] == "start\tend\tspeaker\ttext"
    assert "Hi there friend" in output
