def process_whisper_out(out: dict) -> dict:
    segment_fields = ['start', 'end', 'text']
    out['segments'] = [{k: x[k] for k in segment_fields} for x in out['segments']]
    return out
