from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timedelta
from typing import List
import bcrypt
import jwt
import asyncio
import redis

# Database setup
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup for rate limiting
redis = redis.Redis(host="localhost", port=6379, db=0)
app = FastAPI()
FastAPILimiter.init(redis)

# Secret key for JWT
SECRET_KEY = "supersecretkey"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)

class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)

class Log(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True, index=True)
    path = Column(String)
    method = Column(String)
    status_code = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    response_time = Column(Integer)  # in milliseconds

# Create tables
Base.metadata.create_all(bind=engine)

# Utility functions
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: SessionLocal = Depends()):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")

# API Routes
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: SessionLocal = Depends()):
    user = db.query(User).filter(User.username == form_data.username).first()
    if user is None or not bcrypt.checkpw(form_data.password.encode("utf-8"), user.password.encode("utf-8")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.id})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", response_model=dict)
async def create_user(username: str, password: str, db: SessionLocal = Depends()):
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    user = User(username=username, password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "User created"}

@app.get("/items/", response_model=List[dict], dependencies=[Depends(RateLimiter(times=5, minutes=1))])
async def read_items(db: SessionLocal = Depends(), user: User = Depends(get_current_user)):
    return db.query(Item).all()

@app.post("/items/", response_model=dict)
async def create_item(name: str, description: str, db: SessionLocal = Depends(), user: User = Depends(get_current_user)):
    item = Item(name=name, description=description)
    db.add(item)
    db.commit()
    db.refresh(item)
    return {"message": "Item created", "item_id": item.id}

@app.put("/items/{item_id}", response_model=dict)
async def update_item(item_id: int, name: str, description: str, db: SessionLocal = Depends(), user: User = Depends(get_current_user)):
    item = db.query(Item).filter(Item.id == item_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    item.name = name
    item.description = description
    db.commit()
    return {"message": "Item updated"}

@app.delete("/items/{item_id}", response_model=dict)
async def delete_item(item_id: int, db: SessionLocal = Depends(), user: User = Depends(get_current_user)):
    item = db.query(Item).filter(Item.id == item_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    db.delete(item)
    db.commit()
    return {"message": "Item deleted"}

# Middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()
    response = await call_next(request)
    process_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    db = SessionLocal()
    log = Log(
        path=request.url.path,
        method=request.method,
        status_code=response.status_code,
        response_time=int(process_time)
    )
    db.add(log)
    db.commit()
    db.close()
    return response
