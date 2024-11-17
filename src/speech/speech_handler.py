class SpeechHandler:
    def __init__(self):
        self.stt_model = self._initialize_stt()
        self.tts_model = self._initialize_tts()
        
    async def speech_to_text(self, audio_input):
        text = await self.stt_model.transcribe(audio_input)
        return text
        
    async def text_to_speech(self, text):
        audio = await self.tts_model.synthesize(text)
        return audio 