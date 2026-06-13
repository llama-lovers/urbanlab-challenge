# Important features
- Easy and fast to implement <br>
- High transcription accuracy <br>
- Supports multi-channel audio <br> 
- Built-in speaker diarization <br> 

# Any questions? You may find the answer below <br>
### How to run the app?
`docker compose up -d --build  `
<br>

### What does the `use_context` parameter do?
Whisper processes audio in 30-second segments (you don’t need to split it manually - this happens automatically). If `use_context` is set to `true`, each subsequent 30-second segment will take into account the context from the previous one.
<br>

### How to run on GPU or CPU?
To choose between GPU and CPU, set approptiate value in .env and then go to the `docker-compose.yaml` file and comment or uncomment the appropriate section of the code.<br>
Keep in mind that Whisper large-v3 requires around 10 GB of VRAM to run smoothly.
<br>

### How to change the Whisper model?
To change the Whisper model, update the appropriate value in the `.env` file.  
<br>

### How to change the language?
For better model accuracy, a single language is selected. If you want to change it, you can do so in the `.env` file.  
<br>

### Can I transcribe multi-channel audio?
Yes, you can. Information about the channels will be included in the output. However, keep in mind that for multi-channel audio, channels are selected based on signal energy. If two people speak at the same time, the channel assignment may be inaccurate.
<br>

### More details
More details, such as the output format and how to send a request, can be found in Swagger.

#### Contact
https://www.linkedin.com/in/jakub-janusz-10bb5a2ba/
