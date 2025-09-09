from fastapi import FastAPI
from routes import router, bind_lifecycle

app = FastAPI()
bind_lifecycle(app)
app.include_router(router)
