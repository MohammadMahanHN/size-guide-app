from pydantic import BaseModel
from typing import Dict, List, Optional, Any

# --- Schemas for Admin Panel (CRUD) ---

class NodeBase(BaseModel):
    label: str
    parent_id: Optional[int] = None
    is_end: bool = False
    formula: Optional[str] = None
    default_child_id: Optional[int] = None
    meta: Optional[Dict[str, Any]] = {}

class NodeCreate(NodeBase):
    pass

# --- Schema for API Responses ---

class Node(NodeBase):
    id: int

    class Config:
        from_attributes = True

# --- Schema for Paginated Response ---
class NodePage(BaseModel):
    total: int
    nodes: List[Node]

# --- Schema for User Input ---

class SelectionInput(BaseModel):
    selections: List[int]
    inputs: Dict[str, float] = {}

