# main.py
from fastapi import FastAPI
from routes import router, bind_lifecycle  # import both

app = FastAPI()

# IMPORTANT: bind the engine lifecycle so workers start on boot
bind_lifecycle(app)

# mount routes
app.include_router(router)
