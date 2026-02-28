# Import all the models, so that Base has them before being
# imported by Alembic
from app.database.base_class import Base
from app.models.user import User
from app.models.chapter import Chapter
from app.models.personalization_setting import PersonalizationSetting
from app.models.translation import Translation
from app.models.chat_session import ChatSession


# This import is needed to register the models with SQLAlchemy
# It ensures that Alembic can detect the models when generating migrations