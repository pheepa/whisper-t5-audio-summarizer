import pika
import os
from pydantic_models import *
from utils.utils import process_whisper_out
from t5_summarizer.model import T5Model
import whisper_timestamped as whisper
import logging

RMQ_USER, RMQ_PASSWORD, RMQ_HOST, RMQ_PORT = os.environ['RMQ_USER'], os.environ['RMQ_PASSWORD'], os.environ['RMQ_HOST'], \
    int(os.environ['RMQ_PORT'])
API_URI = os.environ['API_URI']
WHISPER_PATH = os.environ['WHISPER_PATH']


whisper_params = {
    "compute_word_confidence": False,
    "remove_empty_words": True,
}


credentials = pika.PlainCredentials(RMQ_USER, RMQ_PASSWORD)
parameters = pika.ConnectionParameters(RMQ_HOST, RMQ_PORT, '/', credentials)
# Create a connection to RabbitMQ
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

# Declare the queues
channel.queue_declare(queue='PendingMeetingsQueue')
channel.queue_declare(queue='InProcessMeetingsQueue')
channel.queue_declare(queue='ReadyMeetingsQueue')

channel.basic_qos(prefetch_count=1)


# Consumer callback function
def callback(ch, method, properties, body):
    # Process the incoming message and get the result
    channel.basic_publish(
        exchange='',
        routing_key='InProcessMeetingsQueue',
        body=body
    )
    meeting = Meeting.parse_raw(body.decode("utf-8"))

    # url = 'https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0012_8k.wav'
    url = f'{API_URI}/static/{meeting.videoNameId}.wav'
    logging.info('---Audio is loading.')
    audio = whisper.load_audio(url)
    logging.info('---Model is loading.')
    w_model = whisper.load_model(WHISPER_PATH, device="cpu")
    logging.info('---Transcribing started.')
    result = whisper.transcribe(w_model, audio, language="en", **whisper_params)
    result_proc = process_whisper_out(result)
    transcription = Transcription(**result_proc)

    t5_model = T5Model()
    t5_result = t5_model.generate(transcription.text)

    ready_message = ReadyMeeting(videoNameId=meeting.videoNameId,
                                 text=transcription.text,
                                 summary=t5_result,
                                 segments=transcription.segments).json().encode()

    # Publish the result to the ResultsQueue
    channel.basic_publish(
        exchange='',
        routing_key='ReadyMeetingsQueue',
        body=ready_message
    )
    # Acknowledge that the message has been processed
    ch.basic_ack(delivery_tag=method.delivery_tag)


# Start consuming messages
channel.basic_consume(queue='PendingMeetingsQueue', on_message_callback=callback)

# Start consuming (blocking operation)
channel.start_consuming()

# Close the connection
connection.close()
