import sys
import os
from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
# FIX: Import `text` for executing raw SQL commands
from sqlalchemy import func, cast, String, text
import math
import pandas as pd
import io
from typing import List, Optional

# --- Import local modules ---
import models
import schemas
from database import SessionLocal, engine, DATABASE_URL

# --- Path Correction for PyInstaller ---
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

templates_dir = resource_path("templates")
templates = Jinja2Templates(directory=templates_dir)

# --- Dependency to get DB session ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Database Seeding Logic (FINAL FIX FOR POSTGRESQL ID SEQUENCE) ---
def init_db(db: Session):
    if db.query(models.Node).count() == 0:
        print("Database is empty, seeding with initial data...")
        
        initial_nodes = [
            models.Node(id=1, label="نوع لباس", parent_id=None),
            models.Node(id=2, label="ورزشی", parent_id=1),
            models.Node(id=3, label="رسمی", parent_id=1), 
            models.Node(id=4, label="کشسان", parent_id=2),
            models.Node(id=5, label="بدون کشسان", parent_id=2),
            models.Node(id=6, label="کت و شلوار", parent_id=3, is_end=True, formula="height * 2 + chest"),
            models.Node(id=7, label="جذب", parent_id=4, is_end=True,
                   formula="weight / ((height/100)**2) * style_factor", meta={"style_factor": 0.95}),
            models.Node(id=8, label="خیلی جذب", parent_id=4, is_end=True,
                   formula="weight / ((height/100)**2) * style_factor", meta={"style_factor": 1.05}),
            models.Node(id=9, label="آزاد", parent_id=5, is_end=True,
                   formula="chest + waist - 10")
        ]
        db.add_all(initial_nodes)
        db.commit()

        node_3 = db.query(models.Node).filter(models.Node.id == 3).first()
        if node_3:
            node_3.default_child_id = 6
            db.commit()
        
        # FINAL FIX: This synchronizes the auto-increment counter (sequence) in PostgreSQL
        # with the manually inserted data. This is crucial for preventing duplicate key errors.
        if db.bind.dialect.name == "postgresql":
            max_id = db.query(func.max(models.Node.id)).scalar()
            # The sequence name is typically tablename_colname_seq
            db.execute(text(f"SELECT setval('nodes_id_seq', {max_id}, true);"))
            db.commit()

        print("Database initialized and sequence reset successfully.")

with SessionLocal() as db:
    init_db(db)

# --- (The rest of your API endpoints remain the same and are correct) ---
def get_children(parent_id: int, db: Session) -> list[models.Node]:
    return db.query(models.Node).filter(models.Node.parent_id == parent_id).all()

def get_path(node_id: int, db: Session) -> list[str]:
    path = []
    current = db.query(models.Node).filter(models.Node.id == node_id).first()
    while current:
        path.insert(0, current.label)
        if current.parent_id is None:
            break
        current = db.query(models.Node).filter(models.Node.id == current.parent_id).first()
    return path

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def read_admin(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.post("/resolve")
def api_resolve(selection: schemas.SelectionInput, db: Session = Depends(get_db)):
    last_id = 1
    if selection.selections:
        last_id = selection.selections[-1]

    node = db.query(models.Node).filter(models.Node.id == last_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Invalid selection")

    path_labels = get_path(last_id, db) if last_id != 1 else ["شروع"]

    if node.is_end and node.formula:
        env = selection.inputs.copy()
        if node.meta:
            env.update(node.meta)
        
        safe_env = {"math": math}
        safe_env.update(env)

        try:
            result = eval(node.formula, {"__builtins__": {}}, safe_env)
            return {
                "path": path_labels,
                "formula": node.formula,
                "inputs": selection.inputs,
                "result": result
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Calculation failed: {str(e)}")

    children = get_children(last_id, db)
    return {
        "path": path_labels,
        "next_options": [schemas.Node.from_orm(c) for c in children],
        "default_child_id": node.default_child_id
    }

@app.get("/api/nodes", response_model=schemas.NodePage)
def get_all_nodes(search: Optional[str] = None, page: int = 1, page_size: int = 25, db: Session = Depends(get_db)):
    query = db.query(models.Node)
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (models.Node.label.like(search_term)) |
            (cast(models.Node.id, String).like(search_term))
        )
    
    total = query.count()
    nodes = query.order_by(models.Node.id).offset((page - 1) * page_size).limit(page_size).all()
    return {"nodes": nodes, "total": total}

@app.post("/api/nodes", response_model=schemas.Node)
def create_node(node: schemas.NodeCreate, db: Session = Depends(get_db)):
    if node.parent_id:
        parent = db.query(models.Node).filter(models.Node.id == node.parent_id).first()
        if not parent:
            raise HTTPException(status_code=400, detail=f"Parent with id {node.parent_id} not found")
    # We don't pass an ID, we let the database generate it
    node_data = node.model_dump()
    db_node = models.Node(**node_data)
    db.add(db_node)
    db.commit()
    db.refresh(db_node)
    return db_node

@app.get("/api/nodes/{node_id}", response_model=schemas.Node)
def get_node(node_id: int, db: Session = Depends(get_db)):
    db_node = db.query(models.Node).filter(models.Node.id == node_id).first()
    if not db_node:
        raise HTTPException(status_code=404, detail="Node not found")
    return db_node

@app.put("/api/nodes/{node_id}", response_model=schemas.Node)
def update_node(node_id: int, node: schemas.NodeCreate, db: Session = Depends(get_db)):
    db_node = db.query(models.Node).filter(models.Node.id == node_id).first()
    if not db_node:
        raise HTTPException(status_code=404, detail="Node not found")
    if node.parent_id:
        parent = db.query(models.Node).filter(models.Node.id == node.parent_id).first()
        if not parent:
            raise HTTPException(status_code=400, detail=f"Parent with id {node.parent_id} not found")
    
    for key, value in node.model_dump().items():
        setattr(db_node, key, value)
    
    db.commit()
    db.refresh(db_node)
    return db_node

@app.delete("/api/nodes/{node_id}")
def delete_node(node_id: int, db: Session = Depends(get_db)):
    db_node = db.query(models.Node).filter(models.Node.id == node_id).first()
    if not db_node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    children_count = db.query(models.Node).filter(models.Node.parent_id == node_id).count()
    if children_count > 0:
        raise HTTPException(status_code=400, detail=f"Cannot delete node. It is a parent to {children_count} other nodes.")

    db.delete(db_node)
    db.commit()
    return {"detail": "Node deleted successfully"}

@app.get("/api/search-parents", response_model=List[schemas.Node])
def search_parents(q: str, db: Session = Depends(get_db)):
    search_term = f"%{q}%"
    return db.query(models.Node).filter(
        models.Node.is_end == False,
        (models.Node.label.like(search_term)) | (cast(models.Node.id, String).like(search_term))
    ).limit(10).all()

@app.get("/api/tree/roots", response_model=List[schemas.Node])
def get_tree_roots(db: Session = Depends(get_db)):
    return db.query(models.Node).filter(models.Node.parent_id == None).all()

@app.get("/api/tree/children/{node_id}", response_model=List[schemas.Node])
def get_tree_children(node_id: int, db: Session = Depends(get_db)):
    return db.query(models.Node).filter(models.Node.parent_id == node_id).order_by(models.Node.id).all()

@app.get("/api/nodes/export")
def export_nodes(db: Session = Depends(get_db)):
    nodes = db.query(models.Node).all()
    if not nodes:
        raise HTTPException(status_code=404, detail="No nodes to export.")
    df_data = []
    for node in nodes:
        node_dict = {c.name: getattr(node, c.name) for c in node.__table__.columns}
        df_data.append(node_dict)
    df = pd.DataFrame(df_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Nodes')
    output.seek(0)
    headers = {'Content-Disposition': 'attachment; filename="nodes_export.xlsx"'}
    return StreamingResponse(output, headers=headers, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


@app.post("/api/nodes/import")
def import_nodes(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an .xlsx file.")
    try:
        df = pd.read_excel(file.file)
        df = df.where(pd.notnull(df), None)
        for index, row in df.iterrows():
            node_data = row.to_dict()
            node_id = node_data.get('id')
            if not node_id: continue
            existing_node = db.query(models.Node).filter(models.Node.id == node_id).first()
            if existing_node:
                for key, value in node_data.items():
                    if hasattr(existing_node, key): setattr(existing_node, key, value)
            else:
                new_node = models.Node(**node_data)
                db.add(new_node)
        db.commit()
        return {"detail": f"Successfully processed {len(df)} rows."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to process Excel file: {str(e)}")

@app.get("/api/tree/path/{node_id}", response_model=List[int])
def get_node_path(node_id: int, db: Session = Depends(get_db)):
    path_ids = []
    current = db.query(models.Node).filter(models.Node.id == node_id).first()
    if not current:
        raise HTTPException(status_code=404, detail="Node not found")
    while current:
        path_ids.insert(0, current.id)
        if current.parent_id is None: break
        current = db.query(models.Node).filter(models.Node.id == current.parent_id).first()
    return path_ids

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

