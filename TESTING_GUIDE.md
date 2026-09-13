# Testing Guide for Bug Fixes

## Quick Start

1. **Activate virtual environment**:
   ```cmd
   venv\Scripts\activate
   ```

2. **Set FFmpeg PATH**:
   ```cmd
   set PATH=%PATH%;A:\VIT-3rd-sem\3rd_sem\EDI_Accion_lab\ffmpeg\bin
   ```

3. **Start the server**:
   ```cmd
   python -m uvicorn app.main:app --reload
   ```

4. **Open the web interface**:
   - Go to: http://127.0.0.1:8000/static/index.html

---

## Test 1: Bug A - Merge Speaker Collapsing

### What We're Testing
Whether the merge step correctly assigns segments to both SPEAKER_00 and SPEAKER_01, or if it collapses all segments to one speaker.

### How to Test
1. **Record a new conversation**:
   - Click "Start Recording"
   - Have a conversation between two people (or simulate doctor-patient)
   - Speak clearly with natural pauses between speakers
   - Record for at least 60 seconds
   - Click "Stop Recording"

2. **Watch the console output** (where uvicorn is running):
   ```
   Look for:
   ========================================
   STEP 3: Running speaker diarization...
   ✓ Diarization complete...
   Detected 2 speaker(s): SPEAKER_00, SPEAKER_01
   
   STEP 4: Merging speaker labels with transcript...
   DEBUG: Diarization turns (28 total):
     Turn 0: SPEAKER_00 [0.00s - 2.50s]
     Turn 1: SPEAKER_01 [2.50s - 5.00s]
     ...
   
   DEBUG: Merging X Whisper segments with diarization:
     Segment 0: [0.11s - 3.50s]
       Text: "Good morning, what brings you here today?"
       Overlap analysis:
         Turn 0 (SPEAKER_00 [0.00-2.50]): 2.39s overlap
         Turn 1 (SPEAKER_01 [2.50-5.00]): 1.00s overlap
       → Assigned to SPEAKER_00 (turn 0, max overlap: 2.39s)
   ```

3. **Check for warnings**:
   ```
   ⚠️ WARNING: Merge collapsed 2 detected speakers into 1 — possible merge bug!
   Missing speakers: {'SPEAKER_00'}
   ```

### Success Criteria
- ✅ Diarization detects 2 speakers
- ✅ Merge produces 2 speakers in output
- ✅ NO warning about collapsed speakers
- ✅ Segments are distributed between both speakers

### If Bug Still Exists
The detailed logs will show:
- Which segments overlapped which turns
- Why SPEAKER_00 (or SPEAKER_01) was never selected
- This tells us exactly where to fix the overlap calculation

---

## Test 2: Bug B - Groq JSON Failures

### What We're Testing
Whether `openai/gpt-oss-20b` successfully refines roles on the first attempt, or if it fails and falls back to `groq/compound`.

### How to Test
1. **Use the recording from Test 1** (should have 15+ segments for best test)

2. **Watch console for Step 6**:
   ```
   STEP 6: LLM role refinement...
   Trying model: openai/gpt-oss-20b
   Model openai/gpt-oss-20b finish_reason: stop
   Model openai/gpt-oss-20b raw response preview: {"roles": ["Doctor", "Patient", ...
   ✓ LLM refinement successful with model: openai/gpt-oss-20b
   ```

### Success Criteria
- ✅ `openai/gpt-oss-20b` succeeds on first attempt
- ✅ `finish_reason` is "stop" (not "length" or error)
- ✅ Response preview shows valid JSON with "roles" key
- ✅ NO fallback to `openai/gpt-oss-120b` or `groq/compound`

### If Bug Still Exists
You'll see:
```
Trying model: openai/gpt-oss-20b
Model openai/gpt-oss-20b finish_reason: error
✗ Model openai/gpt-oss-20b returned empty content
  Full response object: {...}
→ Trying next fallback model...
```

The logs will show:
- What finish_reason was returned
- Whether content is empty or has data
- What JSON structure (if any) was returned
- Which validation failed

---

## Test 3: Text Cleanup Feature

### What We're Testing
Whether garbled/repeated text is detected and cleaned while preserving the original.

### How to Test

#### Option A: Natural Test (may or may not trigger)
1. Record a normal conversation
2. Check console for:
   ```
   STEP 2.5: Checking for garbled segments...
   No garbled segments detected
   ```
   OR
   ```
   STEP 2.5: Checking for garbled segments...
   ⚠️ 1 segment(s) flagged as likely garbled
   [12.50s] Reason: Repeated word pattern: 'okay'
   Text: okay okay okay okay...
   ```

#### Option B: Simulate Garbled Text (manual test)
Since it's hard to make Whisper hallucinate on demand, you can:

1. **Edit a saved session JSON** to add a garbled segment:
   ```json
   {
     "start": 10.0,
     "end": 12.0,
     "text": "Don't. Don't. Don't. Don't. Don't.",
     "speaker": "SPEAKER_00",
     "role": "Doctor"
   }
   ```

2. **Or test the detection function directly**:
   ```python
   from app.services.text_cleanup import detect_garbled_segments
   
   segments = [
       {"start": 0, "end": 2, "text": "Hello, how are you?"},
       {"start": 2, "end": 4, "text": "Don't. Don't. Don't. Don't."},
       {"start": 4, "end": 6, "text": "I'm feeling sick"}
   ]
   
   result = detect_garbled_segments(segments)
   print(result[1])  # Should have likely_garbled: True
   ```

### Success Criteria
- ✅ Garbled segments are detected with correct reason
- ✅ Cleanup runs only on flagged segments
- ✅ Original text preserved in `text` field
- ✅ Cleaned text in `text_cleaned` field
- ✅ Flag `text_cleanup_applied: true` is set

### UI Testing
1. Open the transcript viewer: http://127.0.0.1:8000/transcript?session=YOUR_SESSION_ID
2. If a segment was cleaned:
   - ✅ See green "CLEANED" badge next to the text
   - ✅ Hover over the badge to see tooltip
   - ✅ Tooltip shows "Original Whisper Output: ..."
   - ✅ Cleaned text is displayed in the bubble

---

## Expected Console Output (Full Pipeline)

```
======================================================================
PROCESSING RECORDING
======================================================================

STEP 1: Loading and preprocessing audio...
✓ Audio loaded: live_recording_20260908_201946.wav
✓ Preprocessing complete

STEP 2: Transcribing with Whisper...
Loading model 'base.en'...
Model loaded successfully!
Transcribing audio: temp_live_processed.wav
[0.11s - 3.50s] Good morning, what brings you here today?
[3.50s - 7.20s] I've been having stomach pain since yesterday.
...
Transcription complete! (17 segments)

STEP 2.5: Checking for garbled segments...
No garbled segments detected

STEP 3: Running speaker diarization...
Running speaker diarization on: temp_live_processed.wav
Constraining to 2-2 speakers
✓ Diarization complete in 12.45s
Detected 2 speaker(s): SPEAKER_00, SPEAKER_01
Generated 28 speaker turn(s)

STEP 4: Merging speaker labels with transcript...

DEBUG: Diarization turns (28 total):
  Turn 0: SPEAKER_00 [0.00s - 2.50s] (duration: 2.50s)
  Turn 1: SPEAKER_01 [2.50s - 5.00s] (duration: 2.50s)
  ...

DEBUG: Unique speakers in diarization: {'SPEAKER_00', 'SPEAKER_01'}

DEBUG: Merging 17 Whisper segments with diarization:
  Segment 0: [0.11s - 3.50s] (3.39s)
    Text: Good morning, what brings you here today?
    Overlap analysis (segment overlaps 2 turn(s)):
      Turn 0 (SPEAKER_00 [0.00-2.50]): 2.39s overlap
      Turn 1 (SPEAKER_01 [2.50-5.00]): 1.00s overlap
    → Assigned to SPEAKER_00 (turn 0, max overlap: 2.39s)
  
  Segment 1: [3.50s - 7.20s] (3.70s)
    Text: I've been having stomach pain since yesterday.
    Overlap analysis (segment overlaps 2 turn(s)):
      Turn 1 (SPEAKER_01 [2.50-5.00]): 1.50s overlap
      Turn 2 (SPEAKER_01 [5.00-7.50]): 2.20s overlap
    → Assigned to SPEAKER_01 (turn 2, max overlap: 2.20s)
  ...

DEBUG: Merge results:
  Diarization detected: 2 speaker(s): {'SPEAKER_00', 'SPEAKER_01'}
  Merge produced: 2 speaker(s): {'SPEAKER_00', 'SPEAKER_01'}
  Segment distribution: {'SPEAKER_00': 8, 'SPEAKER_01': 9}

✓ Merged 17 segments

STEP 5: Mapping speakers to roles...
Mapping SPEAKER_00 → Doctor
Mapping SPEAKER_01 → Patient
✓ Roles assigned

STEP 6: LLM role refinement...
Sending transcript to Groq LLM for role refinement...
Trying model: openai/gpt-oss-20b
Model openai/gpt-oss-20b finish_reason: stop
Model openai/gpt-oss-20b raw response preview: {"roles": ["Doctor", "Patient", "Patient", ...
✓ LLM refinement successful with model: openai/gpt-oss-20b
✓ LLM refined roles: 3 change(s), 14 confirmed
Role distribution: {'Doctor': 8, 'Patient': 9}

✓ Session saved to: output\session_20260908_201946.json
======================================================================
```

---

## Troubleshooting

### Diarization Fails
- Check HF_TOKEN in `.env`
- Verify you accepted terms at https://huggingface.co/pyannote/speaker-diarization-3.1
- Pipeline will fall back to UNKNOWN (not a critical error)

### Groq Fails
- Check GROQ_API_KEY in `.env`
- Verify model name is correct: `openai/gpt-oss-20b`
- Pipeline will keep original diarization roles (not critical)

### Text Cleanup Doesn't Run
- Only runs if segments are flagged as garbled
- Check if Groq is configured
- If no garbled segments detected, feature is working (nothing to clean)

---

## Quick Regression Test

After confirming the fixes work:

1. Record 5 different conversations
2. For each, check:
   - ✅ Both speakers appear in merge output
   - ✅ LLM refinement succeeds with primary model
   - ✅ No crashes or errors
   - ✅ Transcript viewer displays correctly

3. If all pass → fixes are solid!
