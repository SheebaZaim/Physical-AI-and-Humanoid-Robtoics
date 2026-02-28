from typing import Optional
from pydantic import BaseModel


# Shared properties
class UserBase(BaseModel):
    email: str
    username: str
    is_active: bool = True
    is_superuser: bool = False
    software_background: Optional[str] = None
    hardware_background: Optional[str] = None
    preferred_depth: Optional[str] = "intermediate"  # beginner, intermediate, advanced
    preferred_hardware_assumptions: Optional[str] = "simulation"  # simulation, real_hardware, both


# Properties to receive via API on creation
class UserCreate(UserBase):
    email: str
    password: str
    username: str
    software_background: Optional[str] = None
    hardware_background: Optional[str] = None


# Properties to receive via API on update
class UserUpdate(UserBase):
    password: Optional[str] = None
    software_background: Optional[str] = None
    hardware_background: Optional[str] = None
    preferred_depth: Optional[str] = None
    preferred_hardware_assumptions: Optional[str] = None


class UserInDBBase(UserBase):
    id: int

    class Config:
        from_attributes = True


# Additional properties to return via API
class User(UserInDBBase):
    pass


# Additional properties stored in DB
class UserInDB(UserInDBBase):
    hashed_password: str