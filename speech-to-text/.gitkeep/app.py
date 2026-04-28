from flask import Flask, request, jsonify, render_template
import speech_recognition as sr
import io
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)


def try_language(audio_bytes, lang_code):
    """Try transcribing with a specific language."""
    try:
        recognizer = sr.Recognizer()
        wav_io = io.BytesIO(audio_bytes)
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)

        # show_all=True gives alternative results too
        text = recognizer.recognize_google(audio_data, language=lang_code)
        return {"lang": lang_code, "transcript": text, "error": None}

    except sr.UnknownValueError:
        return {"lang": lang_code, "transcript": None, "error": "Could not understand"}
    except sr.RequestError as e:
        return {"lang": lang_code, "transcript": None, "error": str(e)}
    except Exception as e:
        return {"lang": lang_code, "transcript": None, "error": str(e)}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    if len(audio_bytes) < 100:
        return jsonify({"error": "Audio too short"}), 400

    # Get selected language mode from form
    lang_mode = request.form.get("lang_mode", "en-US")

    # Define language groups for code-mixed speech
    lang_groups = {
        "en-US":        ["en-US"],
        "hi-IN":        ["hi-IN"],
        "gu-IN":        ["gu-IN"],
        "hinglish":     ["hi-IN", "en-IN", "en-US"],       # Hindi-English mix
        "guj-eng":      ["gu-IN", "en-IN", "en-US"],       # Gujarati-English mix
        "auto":         ["en-US", "hi-IN", "gu-IN", "en-IN"],  # Try all
    }

    languages = lang_groups.get(lang_mode, ["en-US"])

    try:
        if len(languages) == 1:
            # Single language - simple
            result = try_language(audio_bytes, languages[0])
            if result["transcript"]:
                return jsonify({
                    "results": [result],
                    "best": result["transcript"]
                })
            else:
                return jsonify({
                    "results": [result],
                    "best": "(could not understand audio)"
                })
        else:
            # Multiple languages - try in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(try_language, audio_bytes, lang)
                    for lang in languages
                ]
                results = [f.result() for f in futures]

            # Filter successful results
            successful = [r for r in results if r["transcript"]]

            if successful:
                # Pick the longest transcript as "best" (usually most complete)
                best = max(successful, key=lambda r: len(r["transcript"]))
                return jsonify({
                    "results": results,
                    "best": best["transcript"],
                    "best_lang": best["lang"]
                })
            else:
                return jsonify({
                    "results": results,
                    "best": "(could not understand audio in any language)"
                })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)