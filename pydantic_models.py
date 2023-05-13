from pydantic import BaseModel


class Segment(BaseModel):
    text: str
    start: float
    end: float


class Transcription(BaseModel):
    text: str
    segments: list[Segment]


class Meeting(BaseModel):
    videoNameId: str


class ReadyMeeting(Meeting):
    summary: str
    text: str
    segments: list[Segment]
