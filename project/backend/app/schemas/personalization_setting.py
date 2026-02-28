from typing import Optional
from pydantic import BaseModel
from datetime import datetime


# Shared properties
class PersonalizationSettingBase(BaseModel):
    user_id: int
    chapter_id: int
    depth_level: str = "intermediate"  # beginner, intermediate, advanced
    hardware_assumptions: str = "simulation"  # simulation, real_hardware, both
    examples_modified: bool = False


# Properties to receive via API on creation
class PersonalizationSettingCreate(PersonalizationSettingBase):
    user_id: int
    chapter_id: int


# Properties to receive via API on update
class PersonalizationSettingUpdate(PersonalizationSettingBase):
    depth_level: Optional[str] = None
    hardware_assumptions: Optional[str] = None
    examples_modified: Optional[bool] = None
    adapted_content: Optional[str] = None


class PersonalizationSettingInDBBase(PersonalizationSettingBase):
    id: int
    adapted_content: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Additional properties to return via API
class PersonalizationSetting(PersonalizationSettingInDBBase):
    pass


# Additional properties stored in DB
class PersonalizationSettingInDB(PersonalizationSettingInDBBase):
    adapted_content: str