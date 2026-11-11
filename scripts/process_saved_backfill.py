"""Process authenticated, locally collected recordings on the GitHub runner."""
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from zipfile import ZipFile

from Crypto.Cipher import AES

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api import icourse
from src.ai import bucketer
from src.ai.summarizer import Summarizer
from src.ai.transcriber import Transcriber
from src.data.database import Database
from src.pipeline.ppt_pipeline import PPTPipeline
from src.runtime.reporter import Reporter
from src.runtime.scheduler import Scheduler


class SavedSlides:
    def __init__(self, folder):
        self.folder = folder
        self.items = json.loads((folder / "slides.json").read_text(encoding="utf-8"))

    def get_ppt_list(self, _course_id, _sub_id):
        return [{"created_sec": item["seconds"], "pptimgurl": None} for item in self.items]

    def read_image(self, item):
        name = self.items[item["page_num"] - 1]["file"]
        return (self.folder / Path(name).name).read_bytes()


def decrypt_bundle(source, target):
    encrypted = source.read_bytes()
    key = base64.b64decode(os.environ.pop("ICOURSE_BACKFILL_TRANSFER_KEY"), validate=True)
    cipher = AES.new(key, AES.MODE_GCM, nonce=encrypted[:12])
    clear = cipher.decrypt_and_verify(encrypted[28:], encrypted[12:28])
    with ZipFile(BytesIO(clear)) as bundle:
        bundle.extractall(target)


def process(folder, rows):
    db = Database()
    reporter = Reporter()
    summarizer = Summarizer()
    transcriber = Transcriber()
    scheduler = Scheduler(reporter=reporter)
    ppt = PPTPipeline(db, scheduler, reporter)
    original_fetch = icourse.fetch_ppt_image
    icourse.fetch_ppt_image = lambda client, item: client.read_image(item)
    failed = []
    try:
        for row in rows:
            sub_id = str(row["sub_id"])
            course_id = row["course_id"]
            db.upsert_course(course_id, row["course_title"], row["lecturer_name"])
            db.insert_lecture(sub_id, course_id, row["sub_title"], row["date"])
            existing = db.get_lecture(sub_id)
            if existing and existing.get("summary") and existing.get("processed_at"):
                print(f"BACKFILL {sub_id} already complete", flush=True)
                continue
            stage = "transcribe"
            try:
                target = folder / sub_id
                audio = target / "audio.flac"
                if audio.is_file():
                    transcript, segments = transcriber.transcribe_video(str(audio))
                else:
                    segments = json.loads((target / "transcript.json").read_text(encoding="utf-8"))
                    transcript = " ".join(s["text"] for s in segments)
                if len(transcript.strip()) < 100:
                    raise ValueError("Insufficient lecture transcript")
                db.update_transcript(sub_id, transcript)
                stage = "ocr"
                slides = SavedSlides(target)
                stats = ppt.submit(slides, course_id, sub_id).drain()
                if stats.failed:
                    raise RuntimeError("Some saved PPT pages failed OCR")
                stage = "summarize"
                prompt, _ = bucketer.assemble(transcript, segments, db.get_done_ppt_pages(sub_id))
                summary, model = summarizer.summarize(
                    row["course_title"], prompt, transcript_chars=len(transcript)
                )
                db.update_summary(sub_id, summary, model)
                db.mark_processed(sub_id)
                db.clear_error(sub_id)
                print(f"BACKFILL {sub_id} complete; transcript={len(transcript)} OCR={stats.done} summary={len(summary)} model={model}", flush=True)
            except Exception as exc:
                failed.append(sub_id)
                db.update_error(sub_id, stage, type(exc).__name__)
                print(f"BACKFILL {sub_id} failed at {stage}: {type(exc).__name__}", flush=True)
    finally:
        icourse.fetch_ppt_image = original_fetch
        scheduler.shutdown()
        db.conn.close()
    if failed:
        raise SystemExit("Backfill failed for lectures: " + ", ".join(failed))


if __name__ == "__main__":
    folder = Path(sys.argv[2])
    decrypt_bundle(Path(sys.argv[1]), folder)
    rows = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    process(folder, [row for row in rows if row["status"] == "collected"])

