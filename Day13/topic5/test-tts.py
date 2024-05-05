def detect_intent_with_texttospeech_response(project_id, session_id, texts,
                                             language_code):
    """Returns the result of detect intent with texts as inputs and includes
    the response in an audio format.

    Using the same `session_id` between requests allows continuation
    of the conversation."""
    import dialogflow_v2 as dialogflow
    session_client = dialogflow.SessionsClient()

    session_path = session_client.session_path(project_id, session_id)
    print('Session path: {}\n'.format(session_path))

    for text in texts:
        text_input = dialogflow.types.TextInput(
            text=text, language_code=language_code)

        query_input = dialogflow.types.QueryInput(text=text_input)

        # Set the query parameters with sentiment analysis
        output_audio_config = dialogflow.types.OutputAudioConfig(
            audio_encoding=dialogflow.enums.OutputAudioEncoding
            .OUTPUT_AUDIO_ENCODING_LINEAR_16)

        response = session_client.detect_intent(
            session=session_path, query_input=query_input,
            output_audio_config=output_audio_config)

        print('=' * 20)
        print('Query text: {}'.format(response.query_result.query_text))
        print('Detected intent: {} (confidence: {})\n'.format(
            response.query_result.intent.display_name,
            response.query_result.intent_detection_confidence))
        print('Fulfillment text: {}\n'.format(
            response.query_result.fulfillment_text))
        # The response's audio_content is binary.
        with open('output.wav', 'wb') as out:
            out.write(response.output_audio)
            print('Audio content written to file "output.wav"')
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "testagent-yqxf-36de31839d78.json"
detect_intent_with_texttospeech_response('testagent-yqxf','12345', ['book a room'], 'en')

			