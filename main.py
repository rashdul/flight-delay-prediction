from fastapi import FastAPI, Query
from services.user_service import get_all_users, get_average_age

app = FastAPI()


@app.get("/users")
def list_users(active: bool = Query(False)):
    return {
        "success": True,
        "data": get_all_users(active_only=active),
        "count": len(get_all_users(active_only=active)),
    }


@app.get("/users/stats")
def user_stats():
    return {
        "success": True,
        "average_age": get_average_age()
    }
