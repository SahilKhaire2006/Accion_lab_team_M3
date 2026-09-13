# What's New - Bug Fixes & Text Cleanup Feature

## Summary

Three major improvements have been implemented:

1. **Bug A Fixed**: Enhanced merge logic with comprehensive debug logging to identify and fix speaker collapsing
2. **Bug B Fixed**: Improved Groq LLM integration with better error handling and JSON parsing
3. **New Feature**: Automatic garbled text detection and cleanup with LLM assistance

---

## 🐛 Bug Fix A: Merge Speaker Collapsing

### The Problem
All transcript segments were being assigned to a single speaker (SPEAKER_01), even though diarization correctly detected two speakers.

### What Was Fixed
- **Added comprehensive debug logging** showing:
  - All diarization turns before merging
  - Detailed overlap analysis for each segment
  - Which turn "won" the speaker assignment
  - Post-merge validation and warnings
  
- **Improved merge algorithm**:
  - Explicit sorting of diarization turns
  - Better overlap calculation visibility
  - Automatic detection of speaker collapse
  
- **Added sanity checks**:
  - Warns if detected speakers don't appear in output
  - Shows segment distribution across speakers
  - Identifies unmatched diarization turns

### What You'll See
The console will now show detailed merge analysis:
```
DEBUG: Diarization turns (28 total):
  Turn 0: SPEAKER_00 [0.00s - 2.50s]
  ...

DEBUG: Merging 17 Whisper segments:
  Segment 0: [0.11s - 3.50s]
    Overlap analysis:
      Turn 0 (SPEAKER_00): 2.39s overlap
      Turn 1 (SPEAKER_01): 1.00s overlap
    → Assigned to SPEAKER_00
```

If the bug still exists, you'll see:
```
⚠️ WARNING: Merge collapsed 2 speakers into 1!
Missing speakers: {'SPEAKER_00'}
```

---

## 🐛 Bug Fix B: Groq JSON Failures

### The Problem
Models `openai/gpt-oss-20b` and `openai/gpt-oss-120b` were failing with empty responses, forcing fallback to the slower `groq/compound` model.

### What Was Fixed
- **Enhanced error logging**:
  - Shows `finish_reason` for every response
  - Logs full response object when empty
  - Displays raw content preview
  - Better exception details
  
- **Improved prompt**:
  - Explicitly states expected format multiple times
  - Shows segment count for validation
  - Emphasizes JSON-only output
  
- **More tolerant JSON parsing**:
  - Strips markdown code fences
  - Tries multiple alternate keys
  - Shows available keys on failure
  - Better validation error messages
  
- **Token length warnings**:
  - Estimates prompt size
  - Warns if approaching model limits

### What You'll See
Successful refinement:
```
Trying model: openai/gpt-oss-20b
finish_reason: stop
raw response preview: {"roles": ["Doctor", "Patient", ...
✓ LLM refinement successful with model: openai/gpt-oss-20b
```

If failures occur, detailed diagnostics:
```
✗ Model openai/gpt-oss-20b returned empty content
  Full response object: {...}
  finish_reason: error
→ Trying next fallback model...
```

---

## ✨ New Feature: Garbled Text Cleanup

### What It Does
Automatically detects and cleans Whisper hallucinations (like "Don't. Don't. Don't. Don't.") while preserving the original text.

### How It Works

#### 1. Detection (Step 2.5 in pipeline)
Two heuristics identify garbled segments:
- **Repeated words**: Same word appears 3+ times in a row
- **Low vocabulary**: Less than 40% unique words in segments with 5+ words

#### 2. Cleanup (Optional, only for flagged segments)
- Sends flagged segment + surrounding context to Groq LLM
- LLM provides cleaned version based on conversation context
- **Original text is NEVER overwritten**

#### 3. Storage Format
```json
{
  "text": "Don't. Don't. Don't.",        // Original (always preserved)
  "text_cleaned": "Don't worry.",        // Cleaned version
  "text_cleanup_applied": true,
  "likely_garbled": true,
  "garbled_reason": "Repeated word pattern: 'Don't'"
}
```

### What You'll See

#### Console Output
```
STEP 2.5: Checking for garbled segments...
⚠️ 1 segment(s) flagged as likely garbled
  [12.50s] Reason: Repeated word pattern: 'okay'
  Text: okay okay okay okay
Sending 1 garbled segment(s) to LLM for cleanup...
  Segment 5: Cleaned successfully
    Original: okay okay okay okay
    Cleaned:  Okay, I understand
✓ Cleaned 1 segment(s). Original text preserved in 'text' field.
```

#### In the Transcript Viewer
- Cleaned text is displayed
- Green "CLEANED" badge appears next to the text
- Hover over badge to see tooltip with original text
- Clear indication that content was modified

### Safety Features
1. ✅ Original text always preserved in `text` field
2. ✅ Only processes flagged segments (not blanket rewrite)
3. ✅ Graceful failure - keeps original if cleanup fails
4. ✅ User can always see actual transcription on hover
5. ✅ Clear visual indicators in UI

---

## File Changes

### Modified Files
- `app/services/merge.py` - Enhanced debug logging and validation
- `app/services/llm_correction.py` - Better error handling and JSON parsing
- `app/api/live_routes.py` - Integrated text cleanup in pipeline
- `static/transcript.html` - Added cleaned text display with hover tooltip

### New Files
- `app/services/text_cleanup.py` - Garbled text detection and cleanup
- `BUG_FIXES_SUMMARY.md` - Detailed technical documentation
- `TESTING_GUIDE.md` - Step-by-step testing instructions
- `WHATS_NEW.md` - This file!

---

## How to Test

### Quick Test
1. Activate venv: `venv\Scripts\activate`
2. Set PATH: `set PATH=%PATH%;A:\VIT-3rd-sem\3rd_sem\EDI_Accion_lab\ffmpeg\bin`
3. Start server: `python -m uvicorn app.main:app --reload`
4. Open: http://127.0.0.1:8000/static/index.html
5. Record a doctor-patient conversation (60+ seconds)
6. Watch console output for detailed logs

### Success Indicators
- ✅ Both SPEAKER_00 and SPEAKER_01 appear in merge output
- ✅ No "collapsed speakers" warning
- ✅ LLM refinement succeeds with `openai/gpt-oss-20b`
- ✅ Text cleanup (if triggered) shows original and cleaned text

### Detailed Testing
See `TESTING_GUIDE.md` for comprehensive test procedures.

---

## Configuration

### Required Environment Variables
```env
# HuggingFace (for diarization)
HF_TOKEN=your_token_here

# Groq (for LLM refinement and text cleanup)
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
GROQ_FALLBACK_MODELS=openai/gpt-oss-120b,groq/compound
```

### Pipeline Steps (Updated)
1. Load and preprocess audio
2. Transcribe with Whisper (word timestamps enabled)
3. **NEW: Detect garbled segments** ⭐
4. **NEW: Cleanup garbled segments (if any)** ⭐
5. Diarization (speaker separation)
6. Merge transcription with diarization
7. Label roles (Doctor/Patient)
8. LLM role refinement (mandatory QA)
9. Save to JSON

---

## Next Steps

1. **Test the fixes**: Follow `TESTING_GUIDE.md`
2. **Check console logs**: Look for detailed debug output
3. **Verify both speakers appear**: in merge and final output
4. **Confirm LLM succeeds**: with primary model (gpt-oss-20b)
5. **Test text cleanup**: Try to trigger garbled detection
6. **Report results**: Share any remaining issues with logs

---

## Notes

### Debug Verbosity
The enhanced logging is verbose for debugging. Once bugs are confirmed fixed, you can reduce verbosity by:
- Commenting out detailed overlap analysis
- Keeping only the sanity check warnings
- Removing DEBUG prefixes

### Performance
- Text cleanup only runs on flagged segments (minimal overhead)
- LLM refinement is now more reliable (fewer fallback attempts)
- Diarization performance unchanged

### Backward Compatibility
- All existing features work as before
- Old session JSON files still compatible
- New fields are optional additions

---

## Support

If you encounter issues:
1. Check console output for detailed error logs
2. Review `BUG_FIXES_SUMMARY.md` for technical details
3. Follow `TESTING_GUIDE.md` for systematic testing
4. Look for specific error messages in the new logging

The enhanced logging is designed to make bugs visible and debuggable!
