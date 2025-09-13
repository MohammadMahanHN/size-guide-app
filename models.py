from sqlalchemy import Boolean, Column, Integer, String, JSON, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class Node(Base):
    __tablename__ = "nodes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    label = Column(String, index=True)
    parent_id = Column(Integer, ForeignKey("nodes.id"), nullable=True)
    is_end = Column(Boolean, default=False)
    formula = Column(String, nullable=True)
    
    default_child_id = Column(Integer, ForeignKey("nodes.id"), nullable=True)
    
    meta = Column(JSON, nullable=True)

    parent = relationship("Node", remote_side=[id], backref="children", foreign_keys=[parent_id])

