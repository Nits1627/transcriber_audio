# VDR Audio Transcription Tool

Professional audio transcription tool designed for Voyage Data Recorder (VDR) analysis in maritime settings. Features advanced noise reduction, speaker diarization, and comprehensive output formats.

## Features

### 🎙️ Audio Enhancement
- **Advanced Noise Reduction** - Removes stationary background noise (engines, ventilation, etc.)
- **High-Pass Filtering** - Eliminates low-frequency rumble
- **Volume Normalization** - Ensures consistent audio levels
- Optimized for ship/marine environments

### 👥 Speaker Diarization
- **Automatic Speaker Identification** - Detects and labels different speakers
- **Neutral Speaker IDs** - Uses "Speaker 1", "Speaker 2" format for objectivity
- **Speaker Statistics** - Shows talk time and percentage for each speaker
- Powered by pyannote.audio

### 📝 Multiple Output Formats
- **Plain Text (.txt)** - Clean transcript with optional speaker labels
- **SubRip (.srt)** - Timestamped subtitles with speaker labels
- **JSON (.json)** - Structured data with segments and metadata
- **Word Document (.docx)** - Professional report with formatting and statistics
- **CSV Index** - Summary of all processed files

### 🚀 Easy to Use
- Upload individual WAV files or ZIP archives
- Real-time progress tracking
- Configurable transcription settings
- One-click download of all results

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/vdr-audio-transcriber.git
cd vdr-audio-transcriber
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Get Hugging Face Token** (for speaker diarization)
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with read access
   - Accept the pyannote model terms at https://huggingface.co/pyannote/speaker-diarization-3.1

## Usage

### Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Basic Workflow

1. **Configure Settings** (in sidebar)
   - Enable/disable noise reduction
   - Adjust enhancement parameters
   - Enable speaker diarization (optional)
   - Select transcription model and language

2. **Upload Audio**
   - Upload individual WAV files, or
   - Upload a ZIP archive containing WAV files

3. **Process**
   - Click "Start Transcription"
   - Monitor real-time progress

4. **Download Results**
   - Download complete package as ZIP
   - Includes all transcripts, reports, and summaries

## Configuration Options

### Audio Enhancement
- **Noise Reduction Strength**: 0.0 - 1.0 (default: 0.7)
- **High-Pass Filter**: 50-200 Hz (default: 80 Hz)
- **Volume Normalization**: On/Off (default: On)

### Speaker Diarization
- **Min/Max Speakers**: Optional constraints
- **Hugging Face Token**: Required for diarization

### Transcription
- **Model Size**: tiny, base, small, medium, large-v2, large-v3
- **Language**: Auto-detect or specify
- **Task**: Transcribe or Translate to English
- **VAD Filter**: Voice Activity Detection

## Output Structure

```
output/
├── transcripts/
│   ├── file1.txt          # Plain text transcript
│   ├── file1.srt          # Subtitle format
│   ├── file1.json         # Structured data
│   ├── file2.txt
│   ├── ...
│   ├── index.csv          # Summary of all files
│   ├── all_transcripts.txt # Combined transcripts
│   └── transcription_report.docx # Professional report
```

## Use Cases

- **VDR Analysis** - Analyze ship bridge recordings
- **Incident Investigation** - Review maritime incidents
- **Training** - Create training materials from recordings
- **Documentation** - Generate written records of verbal communications
- **Compliance** - Meet regulatory documentation requirements

## Technical Details

### Core Technologies
- **Streamlit** - Web interface
- **faster-whisper** - Speech recognition
- **pyannote.audio** - Speaker diarization
- **noisereduce** - Audio enhancement
- **python-docx** - Document generation

### Performance
- Processes ~1 minute of audio in 5-15 seconds (depending on model)
- Speaker diarization adds ~5-15 seconds per minute
- Supports files of any length (limited by available RAM)

## Security Considerations

> **Important**: VDR data often contains sensitive information

- Use private deployment for confidential data
- Consider adding authentication for production use
- Implement data retention policies
- Review deployment guide for security options

## Troubleshooting

### Common Issues

**"No module named 'pyannote'"**
```bash
pip install pyannote.audio torch torchaudio
```

**"CUDA not available" warnings**
- Normal on CPU-only systems
- GPU acceleration optional but recommended for large batches

**Out of memory errors**
- Use smaller model (tiny, base, small)
- Process files individually instead of batch
- Increase system RAM

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- OpenAI Whisper team for the speech recognition model
- pyannote.audio team for speaker diarization
- Streamlit team for the web framework

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions

---

**Note**: This tool is designed for professional maritime use. Always verify transcriptions for accuracy, especially in safety-critical applications.
