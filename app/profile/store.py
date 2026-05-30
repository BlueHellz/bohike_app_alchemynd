from tinydb import TinyDB,Query;from pathlib import Path;import os,datetime
Path(os.getenv("PROFILE_DB","./data/profiles.db")).parent.mkdir(parents=True,exist_ok=True)
_db=TinyDB(os.getenv("PROFILE_DB","./data/profiles.db"));_sess=_db.table("sessions");U=Query()
async def save_session(user_id,session_id,blueprint,rating,notes=""):
    _sess.insert({"user_id":user_id,"session_id":session_id,
        "ts":datetime.datetime.now(datetime.UTC).isoformat(),
        "blueprint":blueprint.model_dump(),"rating":rating,"notes":notes})
def get_user_sessions(user_id,limit=50):
    rows=_sess.search(U.user_id==user_id)
    return sorted(rows,key=lambda r:r["ts"],reverse=True)[:limit]
